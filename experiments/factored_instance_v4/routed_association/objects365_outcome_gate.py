"""Fit an outcome readout of the frozen Objects365 gate representation."""
import argparse
from collections import Counter
from itertools import groupby
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from admission_eval import counts, format_counts, frontier, passed, summarize
from objects365_data import FOLDER, HERE, INDEX, digest, read, save_json
from objects365_fit import collect, qualified, replay_metrics
from objects365_rare import RARE_INDEX

OUT = FOLDER / "instance_gate" / "outcomes"
PLAN = HERE / "OBJECTS365_OUTCOME_GATE.md"
EPOCHS = 100
SEED = 20261111
STATES = ("harm", "unsupported", "semantic", "gain")


def state(o):
    if o["target"] < 0:
        return -1
    if o["wrong"] or any(o["lost"]):
        return 0
    if not o["correct"]:
        return 1
    return 3 if any(o["new"]) else 2


@torch.no_grad()
def latent(backbone, x):
    return torch.cat([backbone.net[:-1](v.float().cuda()) for v in x.split(4096)])


def materialize(records, backbone):
    x = latent(backbone, torch.cat([r["x"][:,0] for r in records]))
    labels = [[state(o) for o in r["outcomes"]] for r in records]
    y = torch.tensor([v for row in labels for v in row],device="cuda")
    image = torch.repeat_interleave(torch.arange(len(records),device="cuda"),
                                    torch.tensor([len(row) for row in labels],device="cuda"))
    weight = torch.zeros(len(y),device="cuda")
    for label in range(len(STATES)):
        take = y==label
        group = torch.bincount(image[take],minlength=len(records))
        assert (group>0).any()
        weight[take] = .25 / int((group>0).sum()) / group[image[take]].float()
    assert bool((y>=0).all()) and abs(float(weight.sum())-1)<1e-5
    return x,y,image,weight,len(records)


def loss_rows(model,data,location,scale):
    x,y,image,weight,n = data
    loss = F.cross_entropy(model((x-location)/scale),y,reduction="none")*weight
    return loss.new_zeros(n).scatter_add_(0,image,loss*n)


def fit(doc):
    OUT.mkdir(exist_ok=True,parents=True)
    assert not (OUT/"qualification.json").exists()
    torch.manual_seed(SEED)
    model, source = qualified()
    raw,_ = collect(doc,"train")
    rare = read(RARE_INDEX)
    extra,_ = collect(dict(rows=rare["rows"],counts=dict(train=len(rare["rows"]))),"train",rare=True)
    monitor,_ = collect(doc,"monitor")
    assert not {r["image"] for r in raw+extra}.intersection(r["image"] for r in monitor)
    train, val = materialize(raw+extra,model),materialize(monitor,model)
    del raw,extra,monitor
    location = train[0].mean(0)
    scale = train[0].std(0).clamp_min(.05)
    head = nn.Linear(train[0].shape[1],4).cuda()
    nn.init.zeros_(head.weight)
    nn.init.zeros_(head.bias)
    opt = torch.optim.Adam(head.parameters(),lr=.05)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS)

    def checkpoint(name):
        path = OUT/f"{name}.pt"
        torch.save(dict(model=head.state_dict(), location=location, scale=scale),path.with_suffix(".part"))
        path.with_suffix(".part").replace(path)

    @torch.no_grad()
    def validation():
        return loss_rows(head,val,location,scale).cpu().numpy()

    initial = validation()
    draws = np.random.default_rng(SEED).integers(0,len(initial),(2000,len(initial)))
    lo,hi = np.percentile(initial[draws].mean(1),[2.5,97.5])
    tol = max(.005,float((hi-lo)/2))
    best, best_epoch = float(initial.mean()),0
    checkpoint("best")
    history = []
    for epoch in range(1,EPOCHS+1):
        opt.zero_grad(set_to_none=True)
        objective = loss_rows(head,train,location,scale).mean() + .005*head.weight.square().sum()
        assert torch.isfinite(objective)
        objective.backward()
        assert all(torch.isfinite(p.grad).all() for p in head.parameters())
        opt.step()
        schedule.step()
        with torch.no_grad():
            train_loss = float(loss_rows(head,train,location,scale).mean())
        valid = float(validation().mean())
        assert math.isfinite(train_loss) and math.isfinite(valid)
        if valid < best:
            best,best_epoch = valid,epoch
            checkpoint("best")
        history.append(dict(epoch=epoch,train=train_loss,monitor=valid,lr=schedule.get_last_lr()[0]))
        if epoch%10==0:
            print(history[-1],flush=True)
    checkpoint("final")
    tail = np.array([r["monitor"] for r in history[-5:]])
    tail_train = np.array([r["train"] for r in history[-5:]])
    slope = float(np.polyfit(np.arange(5),tail,1)[0])
    plateau = float(abs(tail_train[-1]-tail_train[0])/tail_train[0])
    checks = dict(schedule=history[-1]["lr"]==0, plateau=plateau<=.01, slope=abs(slope)<=tol/4,
                  swing=float(np.ptp(tail))<=tol, best_gap=float(tail[-1])-best<=tol)
    save_json(OUT/"history.json",history)
    receipt = dict(passed=all(checks.values()),checks=checks,plan=digest(PLAN),index=digest(INDEX),
                   rare_index=digest(RARE_INDEX),backbone=source["checkpoint"],checkpoint=digest(OUT/"best.pt"),
                   final_checkpoint=digest(OUT/"final.pt"),best=best,best_epoch=best_epoch,final=float(tail[-1]),
                   initial=float(initial.mean()),tolerance=tol,plateau=plateau,slope=slope,
                   states={key:torch.bincount(d[1],minlength=4).tolist() for key,d in (("train",train),("monitor",val))})
    save_json(OUT/"qualification.json",receipt)
    print(receipt,flush=True)


@torch.inference_mode()
def first(records,head,backbone,location,scale):
    chosen = []
    observed = Counter()
    for r in records:
        if "actions" not in r:
            continue
        x = latent(backbone,r["x"].flatten(0,1))
        p = head((x-location)/scale).softmax(-1).reshape(len(r["actions"]),3,4)
        mean = p.mean(1)
        mixture_h = -(mean*mean.clamp_min(1e-7).log()).sum(1)
        within_h = -(p*p.clamp_min(1e-7).log()).sum(2).mean(1)
        js = (mixture_h-within_h).clamp_min(0)
        utility = (mean[:,3]-mean[:,0])*(1-js/math.log(4))
        i = int(utility.argmax())
        o = r["outcomes"][i]
        observed.update(first=1,correct=o["correct"],gain=state(o)==3,harm=state(o)==0,unknown=state(o)<0)
        if utility[i] <= 0:
            continue
        q,c,slot = map(int,r["actions"][i])
        chosen.append(dict(o,image=r["image"],query=q,proposed=c,slot=slot,score=float(utility[i])))
    return chosen,dict(observed)


def calibrate(rows):
    current = counts([])
    gains = 0
    best = dict(threshold=None, gain=0, **format_counts(current))
    for score,tied in groupby(sorted(rows,key=lambda r:-r["score"]),key=lambda r:r["score"]):
        options = list(tied)
        current.update(counts(options))
        gains += sum(state(o)==3 for o in options)
        if current["wrong"] or any(current[f"lost_{i}"] for i in range(6)):
            break
        point = format_counts(current)
        if (gains,point["correct"]) > (best["gain"],best["correct"]):
            best = dict(threshold=score,gain=gains,**point)
    return best


def evaluate(doc):
    receipt = read(OUT/"qualification.json")
    assert receipt["passed"] and receipt["index"]==digest(INDEX) and receipt["rare_index"]==digest(RARE_INDEX)
    assert receipt["plan"]==digest(PLAN) and receipt["checkpoint"]==digest(OUT/"best.pt")
    backbone, source = qualified()
    assert source["checkpoint"]==receipt["backbone"]
    data = torch.load(OUT/"best.pt",map_location="cuda",weights_only=True)
    head = nn.Linear(len(data["location"]),4).cuda().eval()
    head.load_state_dict(data["model"])
    params = head,backbone,data["location"],data["scale"]
    policy_rows,_ = collect(doc,"policy")
    rows,cal_first = first(policy_rows,*params)
    cutoff = calibrate([r for r in rows if r["target"]>=0])
    save_json(OUT/"policy.json",dict(qualification=digest(OUT/"qualification.json"),cutoff=cutoff,first=cal_first))
    records,supply = collect(doc,"eval")
    rows,eval_first = first(records,*params)
    selected = [r for r in rows if passed(r["score"],cutoff["threshold"])]
    ap,classes = replay_metrics(doc,records,dict(outcomes=selected))
    report = dict(images=supply["images"],annotated=classes,cal_first=cal_first,first=eval_first,
                  counts=summarize(selected),cutoff=cutoff,eligible=len(rows),unknown=sum(r["target"]<0 for r in selected),
                  gains=sum(state(r)==3 for r in selected),frontier=frontier(rows,[],(0,1,2),True),
                  gain_frontier=calibrate(rows),ap=ap,
                  harm=[r for r in selected if state(r)==0])
    save_json(OUT/"result.json",report)
    print(report,flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode",choices=("train","eval"))
    args = parser.parse_args()
    doc = read(INDEX)
    (fit if args.mode=="train" else evaluate)(doc)
