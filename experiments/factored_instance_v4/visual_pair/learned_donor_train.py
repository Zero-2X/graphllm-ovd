"""Train the complete-instance matcher and check optimization qualification."""
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from learned_donor_cache import ROOT, SHARDS
from dense_donor import save_json

SEED = 20261110
EPOCHS = 36
LR = .0015
BATCH = 512
WIDTH = 64


class DonorMatcher(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = nn.Sequential(nn.Linear(256,WIDTH),nn.LayerNorm(WIDTH))
        self.language = nn.Sequential(nn.Linear(256,WIDTH),nn.LayerNorm(WIDTH))
        self.extent = nn.Sequential(nn.Linear(2*WIDTH,WIDTH),nn.GELU(),nn.Linear(WIDTH,1))
        self.match = nn.Sequential(nn.Linear(3*WIDTH+2,WIDTH),nn.GELU(),nn.Linear(WIDTH,1))

    def regions(self, roi):
        return F.normalize(self.visual(roi.float()),dim=-1)

    def cue(self, text):
        return F.normalize(self.language(text.float()),dim=-1)

    def extent_logit(self, receiver):
        middle = receiver.mean(1)
        spread = receiver.std(1,unbiased=False)
        return self.extent(torch.cat((middle,spread),dim=1)*WIDTH**.5).squeeze(1)

    def compare(self, source, receiver, cue):
        middle = receiver.mean(1)
        extent = self.extent_logit(receiver)
        sim = receiver @ source.transpose(1,2)
        gathered = (sim/.12).softmax(2) @ source
        local = (receiver*gathered).mean(1)
        sender, candidate = source.mean(1)*cue, middle*cue
        coverage = torch.stack((sim.amax(2).mean(1),sim.amax(1).mean(1)),dim=1)
        relation = self.match(torch.cat((WIDTH*torch.cat((local,sender,candidate),dim=1),coverage),dim=1))
        return extent,relation.squeeze(1)

    def forward(self, donor, receiver, text):
        return self.compare(self.regions(donor),self.regions(receiver),self.cue(text))

    @staticmethod
    def posterior(extent, match):
        whole, same = extent.sigmoid(), match.sigmoid()
        return torch.stack((1-whole,whole*(1-same),whole*same),dim=1)


def balanced_weights(groups):
    weight = np.ones(len(groups),dtype=np.float64)
    for k in range(1,groups.shape[1]):
        children = np.unique(groups[:,:k+1],axis=0)
        _, counts = np.unique(children[:,:k],axis=0,return_counts=True)
        _, parent = np.unique(groups[:,:k],axis=0,return_inverse=True)
        weight /= counts[parent]
    _, group, counts = np.unique(groups,axis=0,return_inverse=True,return_counts=True)
    weight /= counts[group]
    return torch.from_numpy(weight/weight.mean()).float()


def read(splits):
    bank, text, rows = [], [], []
    offset = 0
    for split in splits:
        for i in range(SHARDS):
            d = torch.load(ROOT/f"learned_donor_cache_{split}_{i}.pt",map_location="cpu",weights_only=True)
            assert d["columns"]==("donor","receiver","class","truth","state","image")
            table = d["pairs"]
            assert len(table)==len(d["text"]) and table[:,:2].max()<len(d["roi"])
            table[:,:2] += offset
            offset += len(d["roi"])
            bank.append(d["roi"])
            text.append(d["text"])
            rows.append(table)
    table = torch.cat(rows)
    states = torch.bincount(table[:,4],minlength=3)
    assert len(states)==3 and states.min()>0
    w = balanced_weights(table[:,[4,2,5,3]].numpy())
    return (torch.cat(bank).cuda(),torch.cat(text).cuda(),table.cuda(),w.cuda()),states.tolist()


def forward(model, data, i):
    bank, text, rows, weight = data
    r = rows[i]
    extent, match = model(bank[r[:,0]],bank[r[:,1]],text[i])
    y = r[:,4]
    whole = y>0
    loss = F.binary_cross_entropy_with_logits(extent,whole.float(),reduction="none")
    loss += whole*F.binary_cross_entropy_with_logits(match,(y==2).float(),reduction="none")
    return loss, weight[i]


@torch.inference_mode()
def evaluate(model, data):
    model.eval()
    measured = []
    for i in torch.arange(len(data[2]),device="cuda").split(BATCH):
        loss,w = forward(model,data,i)
        measured.append((loss*w).cpu())
    values = torch.cat(measured).numpy()
    assert np.isfinite(values).all()
    return values


def bootstrap_tolerance(ids, losses, weight):
    _, group = np.unique(ids,return_inverse=True)
    l = np.bincount(group,weights=losses)
    w = np.bincount(group,weights=weight)
    rng = np.random.default_rng(SEED)
    samples = rng.integers(0,len(w),size=(2000,len(w)))
    means = l[samples].sum(1)/w[samples].sum(1)
    lo,hi = np.percentile(means,[2.5,97.5])
    return max(.005,float((hi-lo)/2))


def save(model, label, epoch, name):
    path = ROOT/f"{name}_{label}.pt"
    torch.save(dict(model=model.state_dict(),epoch=epoch),path.with_suffix(".part"))
    path.with_suffix(".part").replace(path)


def main(model_type=DonorMatcher, name="learned_donor", plan="LEARNED_DONOR_PLAN_ZH.md",
         epochs=EPOCHS, sampler=None):
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    assert (ROOT/plan).is_file()
    assert not (ROOT/f"{name}_qualification.json").exists()
    train, ntrain = read(("train","mine_train"))
    cal, ncal = read(("cal",))
    assert set(train[2][:,5].unique().tolist()).isdisjoint(cal[2][:,5].unique().tolist())
    model = model_type().cuda()
    epoch_sampler = sampler(train[2].cpu().numpy()) if sampler else None
    initial = evaluate(model,cal)
    tol = bootstrap_tolerance(cal[2][:,5].cpu().numpy(),initial,cal[3].cpu().numpy())
    init_nll = float(initial.mean())
    print("locked training",ntrain,ncal,"initial",init_nll,"tolerance",tol,flush=True)
    optimizer = torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=.02)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,epochs,eta_min=.025*LR)
    history = []
    best = float("inf")
    for epoch in range(1,epochs+1):
        model.train()
        loss_sum = torch.zeros((),device="cuda")
        if epoch_sampler is None:
            indices = torch.randperm(len(train[2]),device="cuda")
            weights = train[3][indices]
        else:
            indices, weights = epoch_sampler.epoch(epoch)
        for batch, weight in zip(indices.split(BATCH),weights.split(BATCH)):
            optimizer.zero_grad(set_to_none=True)
            per_row, _ = forward(model,train,batch)
            loss = (per_row*weight).mean()
            assert torch.isfinite(loss)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
            loss_sum += loss.detach()*len(batch)
        schedule.step()
        value = float(evaluate(model,cal).mean())
        if value<best:
            best,best_epoch = value,epoch
            save(model,"best",epoch,name)
        history.append(dict(epoch=epoch,train=float(loss_sum/len(indices)),cal=value,
                            lr=optimizer.param_groups[0]["lr"]))
        save_json(ROOT/f"{name}_history.json",history)
        print(history[-1],flush=True)
    save(model,"final",epochs,name)
    train_last = np.array([a["train"] for a in history[-5:]])
    cal_last = np.array([a["cal"] for a in history[-5:]])
    plateau = float(abs(train_last[:2].mean()-train_last[-2:].mean())/train_last[:2].mean())
    slope = float(abs(np.polyfit(np.arange(5),cal_last,1)[0]))
    checks = dict(schedule=abs(history[-1]["lr"]-LR*.025)<1e-9,finite=True,plateau=plateau<=.025,
                  cal_slope=slope<=tol/4,cal_spread=float(np.ptp(cal_last))<=tol,
                  cal_gap=float(cal_last[-1]-best)<=tol)
    result = dict(passed=all(checks.values()),checks=checks,train_states=ntrain,cal_states=ncal,
                  initial_cal=init_nll,best_cal=best,final_cal=float(cal_last[-1]),
                  best_epoch=best_epoch,plateau=plateau,cal_slope=slope,cal_tolerance=tol)
    if epoch_sampler is not None:
        result["sampling"] = epoch_sampler.summary
    save_json(ROOT/f"{name}_qualification.json",result)
    print("learned donor qualification",result,flush=True)


if __name__=="__main__":
    main()
