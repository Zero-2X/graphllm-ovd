"""Fit one shared claim reader from matched proposals and same-donor rivals."""
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from competition_cache import FEATURES, KEYS, ROOT, SHARDS, save_json
from learned_donor_train import SEED

NAME = "competition"
EPOCHS = 24
RIDGE = .01


class ClaimReader(nn.Module):
    def __init__(self):
        super().__init__()
        self.readout = nn.Linear(len(FEATURES),1)
        self.register_buffer("mean",torch.zeros(len(FEATURES)))
        self.register_buffer("scale",torch.ones(len(FEATURES)))

    def normalize(self,x):
        return ((x-self.mean)/self.scale).clamp(-6,6)

    def forward(self,x):
        return self.readout(self.normalize(x)).squeeze(-1)

    def score(self,x):
        p = self(x).sigmoid()
        old,new = p.unbind(dim=1)
        return (new*(1-old)).masked_fill(new<=old,0),p


@dataclass
class Claims:
    x: np.ndarray
    y: np.ndarray
    keys: np.ndarray


def load_claims(splits):
    pieces = []
    for split in splits:
        for i in range(SHARDS):
            with np.load(ROOT/f"competition_{split}_{i}.npz") as raw:
                assert tuple(raw["fields"])==FEATURES and tuple(raw["key_fields"])==KEYS
                pieces.append(Claims(raw["evidence"],raw["labels"],raw["keys"]))
    return Claims(*(np.concatenate([getattr(part,k) for part in pieces]) for k in ("x","y","keys")))


def groups(keys):
    order = np.lexsort((keys[:,2],keys[:,1],keys[:,0]))
    breaks = np.flatnonzero(np.any(keys[order[1:]]!=keys[order[:-1]],axis=1))+1
    return np.split(order,breaks)


def paired_indices(keys,y):
    index,weights,summary = [],[],dict(groups=0,mixed_groups=0,multiple_positive=0)
    for group in groups(keys[:,[0,5,3]]):
        known = group[y[group]>=0]
        if not len(known):
            continue
        summary["groups"]+=1
        positive = known[y[known]>0]
        negative = known[y[known]==0]
        if len(positive)>=2:
            summary["multiple_positive"]+=1
        if len(positive) and len(negative):
            summary["mixed_groups"]+=1
            pairs = np.column_stack((np.repeat(positive,len(negative)),np.tile(negative,len(positive))))
            index.append(pairs)
            weights.append(np.full(len(pairs),1/len(pairs)))
    index = np.concatenate(index)
    weight = np.concatenate(weights)
    iid = keys[index[:,0],0]
    _,inv = np.unique(iid,return_inverse=True)
    weight /= np.bincount(inv,weights=weight)[inv]
    summary.update(pairs=len(index),mixed_images=len(np.unique(iid)))
    return index,weight,summary


def design(data,model):
    known = data.y>=0
    values = torch.from_numpy(data.x).cuda()
    x = model.normalize(values)
    ids,inv = np.unique(data.keys[:,0],return_inverse=True)
    ni = len(ids)
    image = np.repeat(inv,2).reshape(-1,2)[known]
    w = 1/np.bincount(image)[image]
    pairs,pair_w,summary = paired_indices(data.keys,data.y[:,1])
    rank_x = x[pairs[:,0],1]-x[pairs[:,1],1]
    rank_image = inv[pairs[:,0]]
    t = lambda a: torch.as_tensor(a,device="cuda")
    d = dict(x=x[known],y=t(data.y[known]).float(),w=t(w).float(),image=t(image),ni=ni,
             rank_x=rank_x,rank_w=t(pair_w).float(),rank_image=t(rank_image))
    summary.update(actions=len(data.keys),images=ni,positives=(data.y==1).sum(0).tolist(),
                   ignored=(data.y<0).sum(0).tolist())
    return d,summary


def pieces(model, d):
    z = model.readout(d["x"]).squeeze(-1)
    point = F.binary_cross_entropy_with_logits(z,d["y"],reduction="none")*d["w"]
    rank = F.softplus(-F.linear(d["rank_x"],model.readout.weight).squeeze(-1))*d["rank_w"]
    return point,rank


@torch.inference_mode()
def monitor(model,d):
    point,rank = pieces(model,d)
    total = torch.zeros(d["ni"],device="cuda")
    total.scatter_add_(0,d["image"],point)
    total.scatter_add_(0,d["rank_image"],rank)
    assert torch.isfinite(total).all()
    return total.cpu().numpy()


def save(model,label,epoch):
    path = ROOT/f"{NAME}_{label}.pt"
    torch.save(dict(model=model.state_dict(),epoch=epoch),path.with_suffix(".part"))
    path.with_suffix(".part").replace(path)


def main():
    torch.manual_seed(SEED)
    assert (ROOT/"COMPETITION_PLAN_ZH.md").is_file()
    assert not (ROOT/f"{NAME}_qualification.json").exists()
    train = load_claims(("train","mine"))
    dev = load_claims(("cal",))
    assert len(train.x)==116773 and len(dev.x)==13661
    assert set(train.keys[:,0]).isdisjoint(dev.keys[:,0])
    assert len(np.unique(train.keys[:,[0,1,3]],axis=0))==len(train.x)
    model = ClaimReader().cuda()
    x = torch.from_numpy(train.x[train.y>=0]).cuda()
    model.mean.copy_(x.mean(0))
    model.scale.copy_(x.std(0).clamp_min(.001))
    t,train_info = design(train,model)
    v,cal_info = design(dev,model)
    with torch.no_grad():
        prevalence = (t["y"]*t["w"]).sum()/t["ni"]
        model.readout.weight.zero_()
        model.readout.bias.copy_(prevalence.logit())
    initial = monitor(model,v)
    rng = np.random.default_rng(SEED)
    sample = rng.integers(0,len(initial),size=(2000,len(initial)))
    lo,hi = np.percentile(initial[sample].mean(1),[2.5,97.5])
    tol = max(.005,float((hi-lo)/2))
    print("claim supervision",train_info,cal_info,"tolerance",tol,flush=True)
    optimizer = torch.optim.LBFGS(model.parameters(),lr=1.,max_iter=4,line_search_fn="strong_wolfe",
                                  tolerance_grad=1e-7,tolerance_change=1e-10)
    def objective():
        p,r = pieces(model,t)
        return (p.sum()+r.sum())/t["ni"]+RIDGE/2*model.readout.weight.square().sum()
    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss = objective()
        assert torch.isfinite(loss)
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters())
        return loss
    history,best = [],float("inf")
    for epoch in range(1,EPOCHS+1):
        optimizer.step(closure)
        cal = float(monitor(model,v).mean())
        loss = float(objective().detach())
        if cal<best:
            best,best_epoch=cal,epoch
            save(model,"best",epoch)
        history.append(dict(epoch=epoch,train=loss,cal=cal))
        save_json(ROOT/f"{NAME}_history.json",history)
        print(history[-1],flush=True)
    save(model,"final",EPOCHS)
    losses = np.array([row["train"] for row in history[-5:]])
    values = np.array([row["cal"] for row in history[-5:]])
    plateau = float(abs(losses[:2].mean()-losses[-2:].mean())/losses[:2].mean())
    slope = float(abs(np.polyfit(np.arange(5),values,1)[0]))
    checks = dict(schedule=len(history)==EPOCHS,finite=True,plateau=plateau<=.025,
                  cal_slope=slope<=tol/4,cal_spread=float(np.ptp(values))<=tol,cal_gap=values[-1]-best<=tol)
    result = dict(passed=bool(all(checks.values())),checks={k:bool(v) for k,v in checks.items()},
                  train=train_info,cal=cal_info,initial_cal=float(initial.mean()),best_cal=best,
                  final_cal=float(values[-1]),best_epoch=best_epoch,plateau=plateau,cal_slope=slope,cal_tolerance=tol)
    save_json(ROOT/f"{NAME}_qualification.json",result)
    print(result,flush=True)


if __name__=="__main__":
    main()
