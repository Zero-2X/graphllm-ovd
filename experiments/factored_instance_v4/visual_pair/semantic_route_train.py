"""Fit the exact same-box action benefit and protection states."""
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

from competition_train import paired_indices
from dense_donor import ROOT, SHARDS, save_json
from learned_donor_train import SEED
from relabel import RelabelGate
from semantic_evidence import FIELDS

NAME = "semantic_route"
EPOCHS = 24
RIDGE = .01
INPUT = tuple(i for i,name in enumerate(FIELDS) if "rival" not in name)
assert len(INPUT)==13


@dataclass
class ActionData:
    rows: np.ndarray
    evidence: np.ndarray
    lost_floor: np.ndarray
    other_native: np.ndarray
    same_query: np.ndarray
    reused: int
    new: int


def read(splits,prefix="semantic_evidence"):
    values = []
    arrays = ("rows","evidence","lost_floor","other_native","same_query")
    for split in splits:
        for i in range(SHARDS):
            with np.load(ROOT/f"{prefix}_{split}_{i}.npz") as raw:
                assert tuple(raw["fields"])==FIELDS
                values.append(ActionData(*(raw[k] for k in arrays),int(raw["reused"]),int(raw["new"])))
    x = ActionData(*(np.concatenate([getattr(v,k) for v in values]) for k in arrays),
                   sum(v.reused for v in values),sum(v.new for v in values))
    assert len(x.rows)==x.reused+x.new and np.isfinite(x.evidence).all()
    assert len(np.unique(x.rows[:,[0,1,3]],axis=0))==len(x.rows)
    return x


def build(data):
    y = np.where((data.rows[:,9]>0)|(data.lost_floor>0),2,data.rows[:,8]).astype(np.int64)
    ids,inv,counts = np.unique(data.rows[:,0],return_inverse=True,return_counts=True)
    w = 1/counts[inv]
    t = lambda x: torch.as_tensor(x,device="cuda")
    state = np.bincount(y,minlength=3)
    assert state.min()>0
    keys = np.column_stack((data.rows[:,:4],np.full(len(y),-1),data.rows[:,4])).astype(np.int64)
    pair,pair_weight,pair_info = paired_indices(keys,(y==1).astype(np.int64))
    d = dict(x=t(data.evidence[:,INPUT]),y=t(y),w=t(w).float(),group=t(inv),images=len(ids),
             pair=t(pair),pair_weight=t(pair_weight).float())
    info = dict(actions=len(y),images=len(ids),states=state.tolist(),reused=data.reused,new=data.new,
                groups=pair_info,fix_lost=int(((data.rows[:,8]>0)&(data.lost_floor>0)).sum()),
                other_native_fix=int(((y==1)&data.other_native).sum()),
                same_query_only_fix=int(((y==1)&data.same_query&~data.other_native).sum()))
    return d,info


@torch.inference_mode()
def monitor(model,data):
    loss = F.cross_entropy(model(data["x"])-model.weight.log(),data["y"],reduction="none")
    values = torch.zeros(data["images"],device="cuda")
    values.scatter_add_(0,data["group"],loss*data["w"])
    assert torch.isfinite(values).all()
    return values.cpu().numpy()


def save(model,label,epoch,name):
    path = ROOT/f"{name}_{label}.pt"
    torch.save(dict(model=model.state_dict(),epoch=epoch,width=len(INPUT)),path.with_suffix(".part"))
    path.with_suffix(".part").replace(path)


def main(name=NAME,prefix="semantic_evidence"):
    torch.manual_seed(SEED)
    assert (ROOT/"SEMANTIC_ROUTE_PLAN_ZH.md").exists()
    assert not (ROOT/f"{name}_qualification.json").exists()
    train,cal = read(("train","mine"),prefix),read(("cal",),prefix)
    assert len(train.rows)==122079 and len(cal.rows)==14325
    assert set(train.rows[:,0]).isdisjoint(cal.rows[:,0])
    t,train_info = build(train)
    v,cal_info = build(cal)
    model = RelabelGate(len(INPUT)).cuda()
    model.mean.copy_(t["x"].mean(0))
    model.scale.copy_(t["x"].std(0).clamp_min(.001))
    freq = torch.zeros(3,device="cuda").scatter_add_(0,t["y"],t["w"])/t["images"]
    weight = freq.rsqrt()
    model.weight.copy_(weight/(freq*weight).sum())
    with torch.no_grad():
        model.readout.bias.copy_((freq*model.weight).log())
    initial = monitor(model,v)
    rng = np.random.default_rng(SEED)
    samples = rng.integers(0,len(initial),size=(2000,len(initial)))
    lo,hi = np.percentile(initial[samples].mean(1),[2.5,97.5])
    tol = max(.005,float((hi-lo)/2))
    print("semantic training",train_info,cal_info,"tolerance",tol,flush=True)
    opt = torch.optim.LBFGS(model.parameters(),lr=1.,max_iter=4,line_search_fn="strong_wolfe",
                            tolerance_grad=1e-7,tolerance_change=1e-10)

    def objective():
        logits = model(t["x"])
        ce = F.cross_entropy(logits,t["y"],weight=model.weight,reduction="none")
        z = logits[:,1]-.5*(logits[:,0]+logits[:,2])
        ranked = z[t["pair"]]
        pair = F.softplus(ranked[:,1]-ranked[:,0])
        return ((ce*t["w"]).sum()+(pair*t["pair_weight"]).sum())/t["images"]+RIDGE/2*model.readout.weight.square().sum()

    def closure():
        opt.zero_grad(set_to_none=True)
        loss = objective()
        assert torch.isfinite(loss)
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters())
        return loss

    history,best = [],float("inf")
    for epoch in range(1,EPOCHS+1):
        opt.step(closure)
        value = float(monitor(model,v).mean())
        if value<best:
            best,best_epoch = value,epoch
            save(model,"best",epoch,name)
        history.append(dict(epoch=epoch,train=float(objective().detach()),cal=value))
        save_json(ROOT/f"{name}_history.json",history)
        print(history[-1],flush=True)
    save(model,"final",EPOCHS,name)
    tr = np.array([x["train"] for x in history[-5:]])
    dv = np.array([x["cal"] for x in history[-5:]])
    plateau = float(abs(tr[:2].mean()-tr[-2:].mean())/tr[:2].mean())
    slope = float(abs(np.polyfit(np.arange(5),dv,1)[0]))
    checks = dict(schedule=len(history)==EPOCHS,finite=True,plateau=plateau<=.025,
                  cal_slope=slope<=tol/4,cal_spread=float(np.ptp(dv))<=tol,cal_gap=float(dv[-1]-best)<=tol)
    checks = {k:bool(v) for k,v in checks.items()}
    result = dict(passed=all(checks.values()),checks=checks,train=train_info,cal=cal_info,
                  initial_cal=float(initial.mean()),best_cal=best,best_epoch=best_epoch,final_cal=float(dv[-1]),
                  cal_tolerance=tol,cal_slope=slope,plateau=plateau)
    save_json(ROOT/f"{name}_qualification.json",result)
    print("semantic qualification",result,flush=True)


if __name__=="__main__":
    main()
