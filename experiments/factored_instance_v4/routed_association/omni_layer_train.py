"""Fit the 2.96M-parameter cross-layer head using OmniLabel supervision."""
import argparse
from collections import Counter
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from omni_layer_cache import FOLDER, INDEX, PLAN, SHARDS, digest, read
from route_screen import save_json


RECEIPT = FOLDER / "qualification.json"
REPAIR_RECEIPT = FOLDER / "qualification_repair.json"
REPAIR_PLAN = PLAN.with_name("OMNI_LAYER_REPAIR_ZH.md")
SEED, EPOCHS, AVERAGE_START = 20261111, 48, 25


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(3584), nn.Linear(3584,768), nn.GELU(), nn.Dropout(.25),
                                 nn.Linear(768,256), nn.GELU(), nn.Dropout(.25), nn.Linear(256,2))

    def forward(self,x):
        return self.net(x)


def load_blocks():
    index = read(INDEX)
    assert index["plan"] == digest(PLAN) and not index["eval_opened"]
    index_sha = digest(INDEX)
    parts = {k:[] for k in ("train","monitor","policy")}
    counts = Counter()
    for shard in range(SHARDS):
        rows = index["rows"][shard::SHARDS]
        for start in range(0,len(rows),32):
            path = FOLDER / f"data_{shard}_{start//32:03d}.pt"
            doc = torch.load(path,map_location="cpu",weights_only=True)
            assert doc["lock"] == dict(index=index_sha,images=[r["id"] for r in rows[start:start+32]]) and not doc["eval_opened"]
            counts.update(doc["counts"])
            for r in doc["records"]:
                assert index["rows"][r["image"]]["split"] == r["split"]
                parts[r["split"]].append(r)
    return parts, counts


def tensorize(records):
    y = torch.tensor([r["target"] for row in records for r in row["outcomes"]],device="cuda")
    x = torch.cat([r["x"][:,0] for r in records]).float().cuda()
    i = torch.repeat_interleave(torch.arange(len(records)),torch.tensor([len(r["actions"]) for r in records])).cuda()
    w = torch.zeros(len(y),device="cuda")
    for label in (0,1):
        take = y == label
        groups = torch.bincount(i[take],minlength=len(records))
        assert (groups>0).any(), label
        w[take] = .5 / int((groups>0).sum()) / groups[i[take]].float()
    assert len(y) == len(x) == len(i) and abs(float(w.sum())-1.)<1e-5
    return dict(x=x,y=y,index=i,weight=w,images=len(records))


@torch.inference_mode()
def losses(model,data):
    model.eval()
    result = torch.zeros(data["images"],device="cuda")
    for start in range(0,len(data["y"]),2048):
        s=slice(start,start+2048)
        value=F.cross_entropy(model(data["x"][s]),data["y"][s],reduction="none")
        result.scatter_add_(0,data["index"][s],value*data["weight"][s]*len(result))
    assert torch.isfinite(result).all()
    return result.cpu().numpy()


def save_model(model,epoch,kind):
    path=FOLDER / f"{kind}.pt"
    torch.save(dict(model=model.state_dict(),epoch=epoch,plan=digest(PLAN)),path.with_suffix(".part"))
    path.with_suffix(".part").replace(path)


def train_epoch(model,data,optimizer):
    model.train()
    order=torch.randperm(len(data["y"]),device="cuda")
    for ids in order.split(128):
        optimizer.zero_grad(set_to_none=True)
        loss=F.cross_entropy(model(data["x"][ids]),data["y"][ids],reduction="none")
        objective=(loss*data["weight"][ids]).sum()*len(order)/len(ids)
        assert torch.isfinite(objective)
        objective.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()


def main():
    assert not RECEIPT.exists()
    torch.manual_seed(SEED)
    parts,counts=load_blocks()
    supply=dict(counts=dict(counts),labeled_images={k:len(v) for k,v in parts.items()},index=digest(INDEX))
    save_json(FOLDER / "supply.json",supply)
    print("OmniLabel training supply",supply,flush=True)
    train,cal=tensorize(parts["train"]),tensorize(parts["monitor"])
    model=Head().cuda()
    average=torch.optim.swa_utils.AveragedModel(model)
    optim=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.03)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optim,EPOCHS)
    initial=losses(model,cal)
    rng=np.random.default_rng(SEED)
    boot=rng.integers(0,len(initial),size=(2000,len(initial)))
    low,high=np.percentile(initial[boot].mean(1),[2.5,97.5])
    tolerance=max(.005,float((high-low)/2))
    best,history=math.inf,[]
    for epoch in range(1,EPOCHS+1):
        train_epoch(model,train,optim)
        schedule.step()
        tracked=model
        if epoch>=AVERAGE_START:
            average.update_parameters(model)
            tracked=average.module
        valid=float(losses(tracked,cal).mean())
        value=float(losses(tracked,train).mean())
        if epoch>=AVERAGE_START and valid<best:
            best,best_epoch=valid,epoch
            save_model(tracked,epoch,"best")
        history.append(dict(epoch=epoch,cal=valid,train=value,lr=schedule.get_last_lr()[0]))
        print(history[-1],flush=True)
        save_json(FOLDER / "history.json",history)
    save_model(average.module,EPOCHS,"final")
    c=np.asarray([r["cal"] for r in history[-5:]])
    t=np.asarray([r["train"] for r in history[-5:]])
    plateau=float(abs(t[:2].mean()-t[-2:].mean())/max(1e-8,t[:2].mean()))
    slope=float(abs(np.polyfit(np.arange(5),c,1)[0]))
    observed_best=min(r["cal"] for r in history)
    checks=dict(schedule=len(history)==EPOCHS and history[-1]["lr"]==0.,finite=True,plateau=plateau<=.025,
                slope=slope<=tolerance/4,spread=float(np.ptp(c))<=tolerance,gap=float(c[-1]-observed_best)<=tolerance)
    receipt=dict(passed=all(checks.values()),checks=checks,parameters=sum(p.numel() for p in model.parameters()),plan=digest(PLAN),
                 index=digest(INDEX),checkpoint=digest(FOLDER/"best.pt"),best_epoch=best_epoch,initial=float(initial.mean()),
                 best=best,observed_best=observed_best,final=float(c[-1]),tolerance=tolerance,supply=supply,eval_opened=False)
    save_json(RECEIPT,receipt)
    print("OmniLabel qualification",receipt,flush=True)


def repair():
    original=read(RECEIPT)
    assert not REPAIR_RECEIPT.exists() and not original["passed"] and not original["eval_opened"]
    assert original["plan"]==digest(PLAN) and original["index"]==digest(INDEX)
    assert original["checkpoint"]==digest(FOLDER/"best.pt")
    previous=read(FOLDER/"history.json")
    anchor=min(previous,key=lambda r:r["cal"])
    assert anchor["epoch"]<AVERAGE_START and anchor["cal"]==original["observed_best"]
    torch.manual_seed(SEED)
    parts,_=load_blocks()
    train,cal=tensorize(parts["train"]),tensorize(parts["monitor"])
    model=Head().cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.03)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,EPOCHS)
    for _ in range(anchor["epoch"]):
        train_epoch(model,train,optimizer)
        schedule.step()
    initial=float(losses(model,cal).mean())
    assert abs(initial-anchor["cal"])<=.003, (initial,anchor)
    save_model(model,anchor["epoch"],"anchor_repair")
    average=torch.optim.swa_utils.AveragedModel(model)
    average.update_parameters(model)
    epochs=16
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=.03)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,epochs)
    best,best_epoch=initial,0
    save_model(average.module,best_epoch,"best_repair")
    history=[]
    for epoch in range(1,epochs+1):
        train_epoch(model,train,optimizer)
        schedule.step()
        average.update_parameters(model)
        valid=float(losses(average.module,cal).mean())
        value=float(losses(average.module,train).mean())
        if valid<best:
            best,best_epoch=valid,epoch
            save_model(average.module,epoch,"best_repair")
        history.append(dict(epoch=epoch,cal=valid,train=value,lr=schedule.get_last_lr()[0]))
        save_json(FOLDER/"history_repair.json",history)
        print(history[-1],flush=True)
    save_model(average.module,epochs,"final_repair")
    c=np.asarray([r["cal"] for r in history[-5:]])
    t=np.asarray([r["train"] for r in history[-5:]])
    tolerance=original["tolerance"]
    plateau=float(abs(t[:2].mean()-t[-2:].mean())/max(1e-8,t[:2].mean()))
    slope=float(abs(np.polyfit(np.arange(5),c,1)[0]))
    reference=min(best,original["observed_best"])
    checks=dict(schedule=len(history)==epochs and history[-1]["lr"]==0.,finite=True,plateau=plateau<=.025,
                slope=slope<=tolerance/4,spread=float(np.ptp(c))<=tolerance,gap=float(c[-1]-reference)<=tolerance)
    receipt=dict(passed=all(checks.values()),checks=checks,parameters=sum(p.numel() for p in model.parameters()),
                 plan=digest(PLAN),repair_plan=digest(REPAIR_PLAN),original=digest(RECEIPT),index=digest(INDEX),
                 checkpoint=digest(FOLDER/"best_repair.pt"),best_epoch=best_epoch,anchor_epoch=anchor["epoch"],
                 initial=initial,best=best,observed_best=reference,final=float(c[-1]),tolerance=tolerance,plateau=plateau,
                 slope=slope,supply=original["supply"],eval_opened=False)
    save_json(REPAIR_RECEIPT,receipt)
    print("OmniLabel repair qualification",receipt,flush=True)


def qualified():
    receipt=REPAIR_RECEIPT if REPAIR_RECEIPT.exists() else RECEIPT
    info=read(receipt)
    path=FOLDER/("best_repair.pt" if receipt==REPAIR_RECEIPT else "best.pt")
    if receipt==REPAIR_RECEIPT:
        assert info["repair_plan"]==digest(REPAIR_PLAN) and info["original"]==digest(RECEIPT)
    assert info["passed"] and not info["eval_opened"] and info["checkpoint"]==digest(path)
    assert info["index"]==digest(INDEX) and info["plan"]==digest(PLAN)
    saved=torch.load(path,map_location="cpu",weights_only=True)
    assert saved["epoch"]==info["best_epoch"] and saved["plan"]==info["plan"]
    model=Head().cuda().eval()
    model.load_state_dict(saved["model"])
    return model,dict(info,receipt=digest(receipt))


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--repair",action="store_true")
    repair() if parser.parse_args().repair else main()
