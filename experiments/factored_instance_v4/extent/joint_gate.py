"""Fit a single factorized gate with supervised within-image action ranking."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from paired_hypotheses import save_json
from train_factor import tolerance

EPOCHS=80
LR=.035
SEED=20261110


def logit(p):
    return p.clamp(1e-4,1-1e-4).logit()


class JointGate(nn.Module):
    def __init__(self,geo_width=3,class_width=3):
        super().__init__()
        self.geo_raw=nn.Parameter(torch.tensor([.54132485]+[-3.]*(geo_width-1)))
        self.geo_bias=nn.Parameter(torch.zeros(()))
        self.classifier=nn.Linear(class_width,3)
        nn.init.zeros_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        self.register_buffer("mean",torch.zeros(class_width))
        self.register_buffer("scale",torch.ones(class_width))
        self.register_buffer("weights",torch.ones(3))

    def forward(self,g,c):
        loc=g@F.softplus(self.geo_raw)+self.geo_bias
        cls=self.classifier(((c-self.mean)/self.scale).clamp(-6,6))
        return loc,cls

    def probabilities(self,g,c):
        loc,cls=self(g,c)
        match=loc.sigmoid()
        s=(cls-self.weights.log()).softmax(-1)
        good=match*s[:,1]
        bad=match*s[:,2]
        return good/(1-good+4*bad).clamp_min(1e-6), torch.stack((good,bad,match),dim=-1)


def design(x,col):
    value=lambda *k: torch.from_numpy(x[:,[col[s] for s in k]].copy()).float()
    sem=value("reject","old_prob","new_prob").clamp_min(1e-5).log()
    c=torch.cat((sem[:,1:]-sem[:,:1],logit(value("native_score"))),dim=-1)
    if "donor_transport" in col:
        cover=logit((1+value("donor_transport"))/2)
        p=torch.cat((logit(value("v5_p50")),cover),dim=-1)
        return p,torch.cat((c,cover),dim=-1)
    p=value("v5_p50","extent","spill")
    p[:,2]=1-p[:,2]
    return logit(p),c


def load(split,prefix="operational"):
    table=[]
    for i in range(4):
        with np.load(prefix+"_"+split+"_"+str(i)+".npz") as data:
            table.append(data["rows"])
            fields=[str(k) for k in data["columns"]]
    return np.concatenate(table),{v:i for i,v in enumerate(fields)}


def dataset(table,fields):
    g,c=design(table,fields)
    a=table[:,fields["aligned"]].astype(bool)
    target=table[:,fields["strict_fix"]]+2*table[:,fields["old_harmed"]]
    assert not ((target>0)&~a).any()
    iid=table[:,fields["image"]].astype(np.int64)
    q=table[:,fields["query"]].astype(np.int64)
    _,geom,inv,count=np.unique(np.stack((iid,q),axis=1),axis=0,return_index=True,return_inverse=True,return_counts=True)
    if "donor_transport" in fields:
        spatial_weight=len(iid)/(len(geom)*count[inv])
        geom=np.arange(len(iid))
    else:
        spatial_weight=np.ones(len(geom))
    ids,group=np.unique(iid,return_inverse=True)
    valid_groups=np.unique(group[target==1])
    payload=dict(g=g,c=c,spatial_weight=torch.from_numpy(spatial_weight).float(),aligned=torch.from_numpy(a),target=torch.from_numpy(target).long(),
                 geom=torch.from_numpy(geom),group=torch.from_numpy(group),has_gain=torch.from_numpy(valid_groups))
    return {k:v.cuda() for k,v in payload.items()}, (iid[geom],iid[a],ids[valid_groups])


def group_lse(score,group,n):
    peak=torch.full((n,),-torch.inf,device=score.device)
    peak.scatter_reduce_(0,group,score.detach(),reduce="amax",include_self=True)
    sums=torch.zeros_like(peak)
    sums.scatter_add_(0,group,(score-peak[group]).exp())
    return peak+sums.clamp_min(1e-30).log()


def pieces(model,data,training):
    loc,cls=model(data["g"],data["c"])
    geom=data["geom"]
    a=data["aligned"]
    spatial=F.binary_cross_entropy_with_logits(loc[geom],a[geom].float(),reduction="none")*data["spatial_weight"]
    corrected=cls-model.weights.log()
    classes=F.cross_entropy(cls[a] if training else corrected[a],data["target"][a],
                            weight=model.weights if training else None,reduction="none")
    p=corrected.softmax(-1)
    good=loc.sigmoid()*p[:,1]
    bad=loc.sigmoid()*p[:,2]
    utility=good.clamp_min(1e-10).log()-(1-good+4*bad).clamp_min(1e-6).log()
    groups=data["group"]
    n=int(groups.max())+1
    truth=data["target"]==1
    within=(group_lse(utility,groups,n)-group_lse(utility[truth],groups[truth],n))[data["has_gain"]]
    return spatial,classes,within


@torch.inference_mode()
def monitor(model,data):
    out=[p.detach().cpu().numpy() for p in pieces(model,data,False)]
    assert all(np.isfinite(x).all() for x in out)
    return out


def summarize(model,data):
    with torch.inference_mode():
        score,p=model.probabilities(data["g"],data["c"])
    utility=score.cpu().numpy()
    groups=data["group"].cpu().numpy()
    y=data["target"].cpu().numpy()
    aligned=data["aligned"].cpu().numpy()
    best={}
    for i,g in enumerate(groups):
        if g not in best or utility[i]>utility[best[g]]:
            best[g]=i
    order=np.array(sorted(best.values(),key=lambda i:-utility[i]),dtype=int)
    clean=(y[order]==1).cumsum()
    spoil=(y[order]==2).cumsum()
    return dict(images=len(order),strict_total=int((y==1).sum()),strict_top1=int(clean[-1]),old_harmed=int(spoil[-1]),
        no_alignment=int((~aligned[order]).sum()),
        first_strict_rank=int(np.flatnonzero(clean)[0]+1) if clean[-1] else None,
        max_strict_zero_old_harm=int(clean[spoil==0].max()) if np.any(spoil==0) else 0,
        top10_strict=int(clean[min(10,len(clean))-1]),top10_old_harmed=int(spoil[min(10,len(clean))-1]),
        mean_align=float(p[:,2].mean()))


def main():
    torch.manual_seed(SEED)
    parser=argparse.ArgumentParser()
    parser.add_argument("--transport",action="store_true")
    options=parser.parse_args()
    source="transport" if options.transport else "operational"
    output="transport_joint" if options.transport else "joint"
    tr,fields=load("train",source)
    mine,mfields=load("mine",source)
    cal,cfields=load("cal",source)
    assert fields==mfields==cfields
    train,_=dataset(np.concatenate((tr,mine)),fields)
    dev,images=dataset(cal,fields)
    assert len(train["target"])==116773 and len(dev["target"])==13661
    model=JointGate(train["g"].shape[1],train["c"].shape[1]).cuda()
    aligned=train["aligned"]
    model.mean.copy_(train["c"][aligned].mean(0))
    model.scale.copy_(train["c"][aligned].std(0).clamp_min(.001))
    counts=torch.bincount(train["target"][aligned],minlength=3)
    assert counts.tolist()==[1833,95,5083],counts
    freq=counts.float()/counts.sum()
    weight=freq.rsqrt()
    model.weights.copy_(weight/(weight*freq).sum())
    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=.02)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS,eta_min=LR*.025)
    before=monitor(model,dev)
    history=[]
    best=float("inf")

    def save(path,epoch):
        tmp=Path(path+".part")
        torch.save(dict(model=model.state_dict(),epoch=epoch,geo_width=train["g"].shape[1],class_width=train["c"].shape[1]),tmp)
        tmp.replace(path)

    for epoch in range(1,EPOCHS+1):
        opt.zero_grad(set_to_none=True)
        l,c,r=pieces(model,train,True)
        loss=l.mean()+c.mean()+.25*r.mean()
        assert torch.isfinite(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        opt.step()
        sched.step()
        after=monitor(model,dev)
        val=[float(x.mean()) for x in after]
        score=val[0]+val[1]+.25*val[2]
        if score<best:
            best=score
            save(output+"_best.pt",epoch)
        record=dict(epoch=epoch,train=float(loss.detach()),cal=score,components=val,lr=opt.param_groups[0]["lr"])
        history.append(record)
        if epoch==1 or epoch%10==0:
            print(record,flush=True)
    save(output+"_final.pt",EPOCHS)
    save_json(output+"_history.json",history)
    t=np.array([h["train"] for h in history[-5:]])
    plateau=float(abs(t[:2].mean()-t[-2:].mean())/t[:2].mean())
    checks=dict(schedule=abs(history[-1]["lr"]-LR*.025)<1e-9,finite=True,plateau=plateau<=.025)
    components={}
    for j,name in enumerate(("alignment","conditional_class_action","within_image")):
        values=np.array([h["components"][j] for h in history])
        tail=values[-5:]
        tol=tolerance(images[j],after[j]-before[j])
        slope=float(abs(np.polyfit(np.arange(5),tail,1)[0]))
        gap=float(values[-1]-values.min())
        spread=float(np.ptp(tail))
        passed=slope<=tol/4 and gap<=tol and spread<=tol
        components[name]=dict(initial=float(before[j].mean()),best=float(values.min()),final=float(values[-1]),
                              tolerance=tol,slope=slope,gap=gap,spread=spread,passed=passed)
    checks["component_stability"]=all(c["passed"] for c in components.values())
    best_state=torch.load(output+"_best.pt",map_location="cpu",weights_only=True)
    model.load_state_dict(best_state["model"])
    detail=summarize(model,dev)
    result=dict(passed=all(checks.values()),checks=checks,components=components,plateau=plateau,
                best_epoch=best_state["epoch"],development=detail,open_fresh_cal=all(checks.values())
                and detail["max_strict_zero_old_harm"]>=2)
    save_json(output+"_qualification.json",result)
    print("joint qualification",result,flush=True)


if __name__=="__main__":
    main()
