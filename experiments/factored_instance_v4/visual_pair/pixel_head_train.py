"""Learn a frozen-query instance mask using target and competing-instance pixels."""
import json

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from dense_donor import ROOT,SHARDS,save_json
from learned_donor_train import SEED
from mask_query_index import COLUMNS

NAME = "pixel_head"
EPOCHS = 24
BATCH = 128
LR = .0015


class PixelHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.query = nn.Sequential(nn.LayerNorm(256),nn.Linear(256,128),nn.GELU(),nn.Linear(128,64))
        self.visual = nn.Sequential(nn.LayerNorm(256),nn.Linear(256,64))
        self.coordinate = nn.Sequential(nn.Linear(2,32),nn.GELU(),nn.Linear(32,64))

    def forward(self,q,p,xy):
        direction = F.normalize(self.query(q.float()),dim=-1)
        pixel = F.normalize(F.gelu(self.visual(p.float())+self.coordinate(xy.float())),dim=-1)
        return 4*(direction[:,None]*pixel).sum(2)


def read(splits):
    fields = ("rows","pixels","query","relative","target","rival")
    banks = []
    for tag in splits:
        for shard in range(SHARDS):
            d = torch.load(ROOT/f"pixel_query_{tag}_{shard}.pt",map_location="cpu",weights_only=True,mmap=True)
            assert d["columns"]==COLUMNS and d["receipt"]["orig_native_exact"]
            banks.append(d)
    data = {k:torch.cat([v[k] for v in banks]) for k in fields}
    labels = data["rows"].numpy().astype(np.int64)
    assert len(labels)==len(np.unique(labels[:,:2],axis=0))
    groups,ann,count = np.unique(labels[:,[0,2]],axis=0,return_inverse=True,return_counts=True)
    ids,group,images = np.unique(labels[:,0],return_inverse=True,return_counts=True)
    image_annotations = np.unique(groups[:,0],return_counts=True)[1]
    w = 1/count[ann]/image_annotations[group]
    info = dict(queries=len(labels),images=len(ids),instances=len(groups),
                rival_queries=int(data.pop("rival").any(1).sum()),
                skipped=sum(v["receipt"]["skipped"] for v in banks))
    data.pop("rows")
    data.update(weight=torch.from_numpy(w).float(),group=torch.from_numpy(group))
    return {k:v.cuda() for k,v in data.items()},info,ids


def losses(model,d,i):
    logits = model(d["query"][i],d["pixels"][i],d["relative"][i])
    y = d["target"][i].float()
    return F.binary_cross_entropy_with_logits(logits,y,reduction="none").mean(1)


@torch.inference_mode()
def monitor(model,d,n):
    model.eval()
    group = torch.zeros(n,device="cuda")
    for i in torch.arange(len(d["target"]),device="cuda").split(BATCH):
        value = losses(model,d,i)
        group.scatter_add_(0,d["group"][i],value*d["weight"][i])
    assert torch.isfinite(group).all()
    return group.cpu().numpy()


def save(model,label,epoch):
    path = ROOT/f"{NAME}_{label}.pt"
    torch.save(dict(epoch=epoch,model=model.state_dict()),path.with_suffix(".part"))
    path.with_suffix(".part").replace(path)


def main():
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    assert (ROOT/"PIXEL_QUERY_PLAN_ZH.md").exists()
    assert not (ROOT/f"{NAME}_qualification.json").exists()
    tr,ti,tids = read(("train","mine"))
    ca,ci,cids = read(("cal",))
    assert set(tids).isdisjoint(cids)
    model = PixelHead().cuda()
    initial = monitor(model,ca,len(cids))
    rng = np.random.default_rng(SEED)
    sample = rng.integers(0,len(cids),size=(2000,len(cids)))
    low,high = np.percentile(initial[sample].mean(1),[2.5,97.5])
    tol = max(.005,float((high-low)/2))
    print("pixel supervision",ti,ci,"tolerance",tol,flush=True)
    opt = torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=.02)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS,eta_min=.025*LR)
    n = len(tr["target"])
    group_scale = n/len(tids)
    history,best = [],float("inf")
    for epoch in range(1,EPOCHS+1):
        model.train()
        total = torch.zeros((),device="cuda")
        for i in torch.randperm(n,device="cuda").split(BATCH):
            opt.zero_grad(set_to_none=True)
            value = losses(model,tr,i)
            weighted = value*tr["weight"][i]
            loss = weighted.mean()*group_scale
            assert torch.isfinite(loss)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            opt.step()
            total += weighted.detach().sum()
        sched.step()
        current = float(monitor(model,ca,len(cids)).mean())
        if current<best:
            best,best_epoch = current,epoch
            save(model,"best",epoch)
        history.append(dict(epoch=epoch,train=float(total/len(tids)),cal=current,lr=opt.param_groups[0]["lr"]))
        save_json(ROOT/f"{NAME}_history.json",history)
        print(history[-1],flush=True)
    save(model,"final",EPOCHS)
    train_last = np.array([x["train"] for x in history[-5:]])
    cal_last = np.array([x["cal"] for x in history[-5:]])
    plateau = float(abs(train_last[:2].mean()-train_last[-2:].mean())/train_last[:2].mean())
    slope = float(abs(np.polyfit(np.arange(5),cal_last,1)[0]))
    checks = dict(schedule=abs(history[-1]["lr"]-.025*LR)<1e-9,finite=True,plateau=plateau<=.025,
                  cal_slope=slope<=tol/4,cal_spread=float(np.ptp(cal_last))<=tol,cal_gap=float(cal_last[-1]-best)<=tol)
    result = dict(passed=all(checks.values()),checks=checks,train=ti,cal=ci,
                  initial_cal=float(initial.mean()),final_cal=float(cal_last[-1]),best_cal=best,best_epoch=best_epoch,
                  plateau=plateau,cal_tolerance=tol,cal_slope=slope)
    save_json(ROOT/f"{NAME}_qualification.json",result)
    print("pixel qualification",result,flush=True)


if __name__=="__main__":
    main()
