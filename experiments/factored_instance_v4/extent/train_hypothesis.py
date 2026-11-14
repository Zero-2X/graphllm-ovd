"""Fit a shared semantic scorer with a third reject state."""
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from paired_hypotheses import save_json
from train_factor import tolerance

SEED=20261110
EPOCHS=24
LR=.001
BATCH=512


class SharedHypothesisHead(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.local=nn.Linear(width+2,1,bias=False)
        self.donor=nn.Linear(width,1,bias=False)
        self.reject=nn.Parameter(torch.tensor(-1.5))
        nn.init.zeros_(self.local.weight)
        nn.init.zeros_(self.donor.weight)

    def forward(self,data,i):
        own=data["own"][i].float()
        delta=data["donor_residual"][i].float()
        native=data["native"][i].float()
        clip=data["clip"][i].float()
        own=F.layer_norm(own,(own.shape[-1],))
        delta=F.layer_norm(delta,(delta.shape[-1],))
        features=torch.cat((own,native.tanh()[...,None],(4*clip)[...,None]),dim=-1)
        loc=self.local(features).squeeze(-1)
        ref=self.donor(delta).squeeze(-1)*data["supported"][i]
        z=.5*native+loc+ref
        return torch.cat((self.reject.expand(len(i),1),z),dim=-1)


@torch.inference_mode()
def validate(head,data):
    logits=[]
    for i in torch.arange(len(data["target"]),device="cuda").split(BATCH):
        logits.append(head(data,i).cpu())
    z=torch.cat(logits)
    loss=F.cross_entropy(z,data["target"].cpu(),reduction="none")
    assert torch.isfinite(z).all() and torch.isfinite(loss).all()
    return z,loss.numpy()


def report(z,data):
    p=z.softmax(-1)
    target=data["target"].cpu()
    utility=(p[:,2]/(1-p[:,2]+4*p[:,1]).clamp_min(1e-6)).numpy()
    eligible=(p[:,2]>p[:,:2].max(1).values).numpy()
    best={}
    old_true=set()
    opportunity=set()
    ids=data["image"].cpu().tolist()
    q=data["query"].cpu().tolist()
    for i,key in enumerate(zip(ids,q)):
        if target[i]==1:
            old_true.add(key)
        if target[i]==2:
            opportunity.add(key)
        if eligible[i] and (key not in best or utility[i]>utility[best[key]]):
            best[key]=i
    picks=list(best.values())
    states=torch.bincount(target[picks],minlength=3).tolist()
    return dict(pairs=len(target),eligible_edits=len(picks),fix=states[2],spoil=states[1],other=states[0],
                originally_correct_queries=len(old_true),queries_with_true_alternative=len(opportunity),
                direct_pair_accuracy=float((z.argmax(1)==target).float().mean()))


def main():
    torch.manual_seed(SEED)
    torch.set_float32_matmul_precision("high")
    splits={split:torch.load("hypothesis_"+split+".pt",map_location="cpu",weights_only=True)
            for split in ("train","cal")}
    tr,ca=[{k:v.cuda() for k,v in splits[s].items()} for s in ("train","cal")]
    width=tr["own"].shape[-1]
    head=SharedHypothesisHead(width).cuda()
    optim=torch.optim.AdamW(head.parameters(),lr=LR,weight_decay=.03)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optim,EPOCHS,eta_min=LR*.025)
    baseline,before=validate(head,ca)
    history=[]
    best=float("inf")

    def save(path,epoch):
        temp=Path(path+".part")
        torch.save(dict(model=head.state_dict(),width=width,epoch=epoch),temp)
        temp.replace(path)

    for epoch in range(1,EPOCHS+1):
        total=torch.zeros((),device="cuda")
        n=len(tr["target"])
        for i in torch.randperm(n,device="cuda").split(BATCH):
            optim.zero_grad(set_to_none=True)
            z=head(tr,i)
            loss=F.cross_entropy(z,tr["target"][i])
            assert torch.isfinite(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True)
            optim.step()
            total+=loss.detach()*len(i)
        scheduler.step()
        _,loss=validate(head,ca)
        value=float(loss.mean())
        if value<best:
            best=value
            save("hypothesis_best.pt",epoch)
        item=dict(epoch=epoch,train=float(total)/n,cal=value,lr=optim.param_groups[0]["lr"])
        history.append(item)
        print(item,flush=True)
    save("hypothesis_final.pt",EPOCHS)
    save_json("hypothesis_history.json",history)
    train=np.array([r["train"] for r in history[-5:]])
    cal=np.array([r["cal"] for r in history[-5:]])
    change=float(abs(train[:2].mean()-train[-2:].mean())/train[:2].mean())
    tol=tolerance(splits["cal"]["image"].numpy(),loss-before)
    slope=float(abs(np.polyfit(np.arange(5),cal,1)[0]))
    checks=dict(schedule=abs(history[-1]["lr"]-LR*.025)<1e-9,finite=True,plateau=change<=.025,
                cal_slope=slope<=tol/4,cal_spread=float(np.ptp(cal))<=tol,cal_gap=float(cal[-1]-best)<=tol)
    state=torch.load("hypothesis_best.pt",map_location="cpu",weights_only=True)
    head.load_state_dict(state["model"])
    val,_=validate(head,ca)
    reverse={k:(v.flip(1) if k in ("own","donor_residual","supported","native","clip") else v) for k,v in ca.items()}
    r,_=validate(head,reverse)
    assert torch.allclose(r[:,[0,2,1]],val,atol=1e-6,rtol=1e-6)
    result=dict(passed=all(checks.values()),checks=checks,initial_cal=float(before.mean()),best_cal=best,
                final_cal=float(cal[-1]),plateau=change,cal_tolerance=tol,cal_slope=slope,best_epoch=state["epoch"],
                development_prior=report(baseline,ca),development_shared_head=report(val,ca))
    save_json("hypothesis_qualification.json",result)
    print("hypothesis qualification",result,flush=True)


if __name__=="__main__":
    main()
