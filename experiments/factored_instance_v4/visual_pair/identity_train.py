"""Fit matchability and conditional identity from labeled native-query pairs."""
import json

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from dense_donor import ROOT,SHARDS,save_json
from identity_features import FEATURES
from identity_pairs import PAIR_FIELDS
from learned_donor_train import SEED

NAME = "identity"
EPOCHS = 24
RIDGE = .01


class IdentityReader(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList((nn.Linear(len(FEATURES),1),nn.Linear(len(FEATURES)-2,1)))
        self.register_buffer("mean",torch.zeros(len(FEATURES)))
        self.register_buffer("scale",torch.ones(len(FEATURES)))
        self.register_buffer("odds_weight",torch.zeros(2))

    def normalize(self,x):
        return ((x-self.mean)/self.scale).clamp(-6,6)

    def logit(self,x,i):
        return self.heads[i](x if i==0 else x[:,:-2]).squeeze(-1)

    def probability(self,x):
        z = self.normalize(x)
        return torch.stack([(self.logit(z,i)-self.odds_weight[i]).sigmoid() for i in range(2)],dim=1)


def read(splits):
    rows,values = [],[]
    for split in splits:
        for shard in range(SHARDS):
            with np.load(ROOT/f"identity_features_{split}_{shard}.npz") as raw:
                assert tuple(raw["columns"])==PAIR_FIELDS+("enriched",)
                assert tuple(raw["fields"])==FEATURES
                rows.append(raw["rows"])
                values.append(raw["features"])
    return np.concatenate(rows),np.concatenate(values)


def prepare(rows,x,model):
    ids,inv = np.unique(rows[:,0].astype(np.int64),return_inverse=True)
    z = model.normalize(torch.from_numpy(x).cuda())
    groups = []
    for mask,labels in ((rows[:,-1]==0,rows[:,7]>0),(rows[:,7]>0,rows[:,7]==1)):
        img = inv[mask]
        weights = 1/np.bincount(img)[img]
        weights *= len(ids)/weights.sum()
        assert len(np.unique(labels[mask]))==2
        groups.append(dict(x=z[mask],y=torch.from_numpy(labels[mask]).float().cuda(),
                           group=torch.from_numpy(img).cuda(),weight=torch.from_numpy(weights).float().cuda()))
    return groups,len(ids)


@torch.inference_mode()
def monitor(model,groups,n):
    loss = torch.zeros(n,device="cuda")
    for i,d in enumerate(groups):
        z = model.logit(d["x"],i)-model.odds_weight[i]
        bce = F.binary_cross_entropy_with_logits(z,d["y"],reduction="none")
        loss.scatter_add_(0,d["group"],bce*d["weight"])
    assert torch.isfinite(loss).all()
    return loss.cpu().numpy()


def save(model,suffix,epoch):
    path = ROOT/f"{NAME}_{suffix}.pt"
    torch.save(dict(model=model.state_dict(),epoch=epoch),path.with_suffix(".part"))
    path.with_suffix(".part").replace(path)


def pair_summary(rows,picked):
    real = picked&(rows[:,-1]==0)
    y = rows[:,7]
    return dict(accepted=int(picked.sum()),same=int((picked&(y==1)).sum()),different=int((picked&(y==2)).sum()),
                unconfirmed=int((picked&(y==0)).sum()),real_accepted=int(real.sum()),
                real_same=int((real&(y==1)).sum()),real_different=int((real&(y==2)).sum()),
                real_unconfirmed=int((real&(y==0)).sum()))


@torch.inference_mode()
def development(model,rows,x):
    p = model.cpu().probability(torch.from_numpy(x)).numpy()
    score = p.prod(1)
    order = np.argsort(-score,kind="stable")
    best,max_clean = None,0
    y = rows[:,7]
    stats = {kind:np.bincount(rows[rows[:,-1]==i,7].astype(int),minlength=3).tolist()
             for i,kind in enumerate(("real","enriched"))}
    for n,i in enumerate(order,1):
        if n<len(order) and score[order[n]]==score[i]:
            continue
        selected = order[:n]
        s = pair_summary(rows,np.isin(np.arange(len(rows)),selected))
        if s["different"]:
            break
        if s["real_accepted"] and s["real_same"]/s["real_accepted"]>=.90:
            max_clean = max(max_clean,s["real_same"])
            if s["real_same"]>=10:
                point = dict(s,threshold=float(score[i]),real_precision=s["real_same"]/s["real_accepted"])
                if best is None or (point["real_same"],-point["accepted"])>(best["real_same"],-best["accepted"]):
                    best = point
    known = y>0
    same = p[known,1][y[known]==1]
    different = p[known,1][y[known]==2]
    auc = float(((same[:,None]>different).sum()+.5*(same[:,None]==different).sum())/(len(same)*len(different)))
    result = dict(stats=stats,known_identity_auc=auc,threshold=best,max_clean_real_same=max_clean,
                  passed=best is not None,scope="old identity calibration; held-out by image")
    np.savez_compressed(ROOT/"identity_development.npz",rows=rows,probability=p,score=score)
    save_json(ROOT/"identity_result.json",result)
    print("identity development",result,flush=True)


def main():
    torch.manual_seed(SEED)
    assert (ROOT/"IDENTITY_PLAN_ZH.md").exists()
    assert not (ROOT/"identity_qualification.json").exists()
    train,x = read(("train","mine"))
    cal,v = read(("cal",))
    assert set(train[:,0]).isdisjoint(cal[:,0])
    model = IdentityReader().cuda()
    xgpu = torch.from_numpy(x).cuda()
    model.mean.copy_(xgpu.mean(0))
    model.scale.copy_(xgpu.std(0).clamp_min(.001))
    t,nt = prepare(train,x,model)
    d,nd = prepare(cal,v,model)
    with torch.no_grad():
        for i,g in enumerate(t):
            freq = (g["weight"]*g["y"]).sum()/nt
            priors = torch.stack((1-freq,freq))
            weight = priors.rsqrt()
            g["class_weight"] = weight/(weight*priors).sum()
            model.odds_weight[i] = (weight[1]/weight[0]).log()
            model.heads[i].weight.zero_()
            model.heads[i].bias.copy_(freq.logit()+model.odds_weight[i])
    initial = monitor(model,d,nd)
    rng = np.random.default_rng(SEED)
    sample = rng.integers(0,len(initial),size=(2000,len(initial)))
    lo,hi = np.percentile(initial[sample].mean(1),[2.5,97.5])
    tol = max(.005,float((hi-lo)/2))
    opt = torch.optim.LBFGS(model.parameters(),lr=1.,max_iter=4,line_search_fn="strong_wolfe",
                            tolerance_grad=1e-7,tolerance_change=1e-10)

    def objective():
        loss = 0.
        for i,g in enumerate(t):
            bce = F.binary_cross_entropy_with_logits(model.logit(g["x"],i),g["y"],reduction="none")
            weights = g["weight"]*g["class_weight"][g["y"].long()]
            loss += (bce*weights).sum()/nt+RIDGE/2*model.heads[i].weight.square().sum()
        return loss

    def closure():
        opt.zero_grad(set_to_none=True)
        loss = objective()
        assert torch.isfinite(loss)
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters())
        return loss

    best,history = float("inf"),[]
    for epoch in range(1,EPOCHS+1):
        opt.step(closure)
        value = float(monitor(model,d,nd).mean())
        if value<best:
            best,best_epoch = value,epoch
            save(model,"best",epoch)
        history.append(dict(epoch=epoch,train=float(objective().detach()),cal=value))
        save_json(ROOT/"identity_history.json",history)
        print(history[-1],flush=True)
    save(model,"final",EPOCHS)
    tx = np.array([a["train"] for a in history[-5:]])
    vx = np.array([a["cal"] for a in history[-5:]])
    plateau = float(abs(tx[:2].mean()-tx[-2:].mean())/tx[:2].mean())
    slope = float(abs(np.polyfit(np.arange(5),vx,1)[0]))
    checks = dict(schedule=len(history)==EPOCHS,finite=True,plateau=plateau<=.025,
                  cal_slope=slope<=tol/4,cal_spread=float(np.ptp(vx))<=tol,cal_gap=float(vx[-1]-best)<=tol)
    receipt = dict(passed=all(checks.values()),checks=checks,train_pairs=len(train),cal_pairs=len(cal),
                   initial_cal=float(initial.mean()),best_cal=best,final_cal=float(vx[-1]),best_epoch=best_epoch,
                   cal_tolerance=tol,cal_slope=slope,plateau=plateau)
    save_json(ROOT/"identity_qualification.json",receipt)
    print("identity qualification",receipt,flush=True)
    if receipt["passed"]:
        payload = torch.load(ROOT/"identity_best.pt",map_location="cpu",weights_only=True)
        model.load_state_dict(payload["model"])
        development(model,cal,v)


if __name__=="__main__":
    main()
