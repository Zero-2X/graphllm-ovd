"""Train and evaluate the full Objects365 MLP on edge-preserving Qwen features."""
import argparse
from collections import Counter
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from objects365_edge_cache import OUT, PLAN, split_document
from objects365_cache import block_lock, split_rows
from objects365_data import BLOCK, FOLDER, HERE, INDEX, SHARDS, digest, read, save_json
from objects365_fit import collect, replay_metrics
from objects365_outcome_gate import calibrate, state
from admission_eval import frontier, passed, summarize

SEED = 20261111
BATCH = 512
ENCODER = HERE/"objects365_instance_gate.py"


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(3584), nn.Linear(3584,768), nn.GELU(), nn.Dropout(.30),
            nn.Linear(768,256), nn.GELU(), nn.Dropout(.30), nn.Linear(256,4))

    def forward(self, features):
        return self.network(features)


def edge_records(doc, split):
    source, original, index = split_document(doc,split)
    encoder_hash, plan_hash = digest(ENCODER),digest(PLAN)
    records, counts = {}, Counter()
    for shard in range(SHARDS):
        rows = split_rows(source,original,shard)
        for offset in range(0,len(rows),BLOCK):
            lock = block_lock(index,rows[offset:offset+BLOCK])
            lock.update(plan=plan_hash,encoder=encoder_hash)
            path = OUT/f"{split}_{shard}_{offset//BLOCK:03d}.pt"
            block = torch.load(path,map_location="cpu",weights_only=True)
            assert block["lock"] == lock
            counts.update(block["counts"])
            for row in block["records"]:
                assert row["image"] not in records
                records[row["image"]] = row
    assert counts["images"] == sum(r["split"]==original for r in source["rows"])
    return records,dict(counts)


def dataset(records):
    features, labels, images = [], [], []
    for number, record in enumerate(records.values()):
        keep = record["y"] >= 0
        features.append(record["x"][keep])
        labels.append(record["y"][keep])
        images.append(torch.full((int(keep.sum()),),number))
    x,y,im = (torch.cat(items).cuda() for items in (features,labels,images))
    weight = torch.zeros(len(y),device="cuda")
    n = len(records)
    for cls in range(4):
        selected = y==cls
        by_image = torch.bincount(im[selected],minlength=n)
        image_count = int((by_image>0).sum())
        assert image_count
        weight[selected] = .25 / image_count / by_image[im[selected]].float()
    assert abs(float(weight.sum())-1)<1e-5
    return x,y,im,weight,n


def view_loss(model, features, labels):
    logits = model(features.float().flatten(0,1))
    return F.cross_entropy(logits,labels.repeat_interleave(2),reduction="none").reshape(-1,2).mean(1)


@torch.no_grad()
def losses(model,data):
    x,y,im,w,n = data
    model.eval()
    per_image = torch.zeros(n,device="cuda")
    for start in range(0,len(y),2048):
        take = slice(start,start+2048)
        values = view_loss(model,x[take],y[take]) * w[take]
        per_image.scatter_add_(0,im[take],values)
    return (per_image*n).cpu().numpy()


def training_step(model,data,optimizer):
    x,y,_,w,_ = data
    model.train()
    order = torch.randperm(len(y),device="cuda")
    for batch in order.split(BATCH):
        optimizer.zero_grad(set_to_none=True)
        objective = (view_loss(model,x[batch],y[batch])*w[batch]).mean()*len(y)
        objective.backward()
        optimizer.step()
    if not all(bool(p.isfinite().all()) for p in model.parameters()):
        raise FloatingPointError("Non-finite MLP weights")


def checkpoint(model,name,epoch):
    path = OUT/f"{name}.pt"
    torch.save(dict(model=model.state_dict(),epoch=epoch),path.with_suffix(".part"))
    path.with_suffix(".part").replace(path)
    return digest(path)


def fit(doc,repair):
    torch.manual_seed(SEED)
    torch.set_float32_matmul_precision("high")
    main, main_counts = edge_records(doc,"train")
    extra, extra_counts = edge_records(doc,"rare_train")
    monitor, monitor_counts = edge_records(doc,"monitor")
    assert not set(main).intersection(extra) and not (set(main)|set(extra)).intersection(monitor)
    train, validation = dataset({**main,**extra}),dataset(monitor)
    del main,extra,monitor
    model = Head().cuda()
    receipt_path = OUT/"qualification.json"
    plan_hash, index_hash = digest(PLAN),digest(INDEX)
    if repair:
        previous = read(receipt_path)
        assert not previous["passed"] and not previous["repair"] and previous["plan"]==plan_hash
        assert previous["checkpoint"]==digest(OUT/"best.pt")
        model.load_state_dict(torch.load(OUT/"best.pt",map_location="cpu",weights_only=True)["model"])
        epochs, lr, average_start, suffix = 30,.00002,1,"_repair"
    else:
        assert not receipt_path.exists()
        epochs, lr, average_start, suffix = 60,.0001,31,""
    optimizer = torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.05)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,epochs)
    average = torch.optim.swa_utils.AveragedModel(model)
    if repair:
        average.update_parameters(model)
    initial = losses(model,validation)
    draw = np.random.default_rng(SEED).integers(0,len(initial),(2000,len(initial)))
    low, high = np.percentile(initial[draw].mean(1),[2.5,97.5])
    tolerance = previous["tolerance"] if repair else max(.005,float((high-low)/2))
    best, best_epoch = float(initial.mean()),0
    best_hash = checkpoint(model,"best"+suffix,0)
    history = []
    for epoch in range(1,epochs+1):
        training_step(model,train,optimizer)
        schedule.step()
        if epoch >= average_start:
            average.update_parameters(model)
        current = average.module if epoch >= average_start else model
        train_loss, valid_loss = float(losses(current,train).mean()),float(losses(current,validation).mean())
        assert math.isfinite(train_loss) and math.isfinite(valid_loss)
        if valid_loss < best:
            best,best_epoch = valid_loss,epoch
            best_hash = checkpoint(current,"best"+suffix,epoch)
        history.append(dict(epoch=epoch,train=train_loss,monitor=valid_loss,lr=schedule.get_last_lr()[0]))
        print(history[-1],flush=True)
    final_hash = checkpoint(average.module,"final"+suffix,epochs)
    val_tail = np.array([row["monitor"] for row in history[-5:]])
    train_tail = np.array([row["train"] for row in history[-5:]])
    slope = float(np.polyfit(np.arange(5),val_tail,1)[0])
    plateau = float(abs(train_tail[-1]-train_tail[0])/train_tail[0])
    checks = dict(schedule=history[-1]["lr"]==0,plateau=plateau<=.025,
                  slope=abs(slope)<=tolerance/4,spread=float(np.ptp(val_tail))<=tolerance,
                  gap=float(val_tail[-1])-best<=tolerance)
    receipt = dict(passed=all(checks.values()),checks=checks,plan=plan_hash,index=index_hash,encoder=digest(ENCODER),
                   checkpoint=best_hash,final_checkpoint=final_hash,model="best"+suffix+".pt",repair=repair,
                   best=best,best_epoch=best_epoch,final=float(val_tail[-1]),initial=float(initial.mean()),
                   tolerance=tolerance,slope=slope,plateau=plateau,parameters=sum(p.numel() for p in model.parameters()),
                   train_counts=main_counts,rare_counts=extra_counts,monitor_counts=monitor_counts,
                   train_states=torch.bincount(train[1],minlength=4).tolist(),
                   monitor_states=torch.bincount(validation[1],minlength=4).tolist())
    save_json(OUT/("repair_history.json" if repair else "history.json"),history)
    if repair:
        save_json(OUT/"initial_qualification.json",previous)
    save_json(receipt_path,receipt)
    print(receipt,flush=True)


@torch.inference_mode()
def rank(records, edge, model):
    first, admissible, audit = Counter(), [], {}
    for record in records:
        if record["image"] not in edge:
            continue
        item = edge[record["image"]]
        assert torch.equal(item["actions"],record["actions"])
        predictions = model(item["x"].float().cuda()).softmax(-1)
        average = predictions.mean(1)
        mixture_h = -(average*average.clamp_min(1e-7).log()).sum(1)
        within_h = -(predictions*predictions.clamp_min(1e-7).log()).sum(2).mean(1)
        disagreement = (mixture_h-within_h).clamp_min(0)
        score = (average[:,3]-average[:,0])*(1-disagreement/math.log(4))
        i = int(score.argmax())
        outcome = record["outcomes"][i]
        strict = int(item["y"][i])
        first.update(total=1,correct=outcome["correct"],gain=state(outcome)==3,
                     strict_correct=strict>=2,strict_gain=strict==3,harm=state(outcome)==0,unknown=outcome["target"]<0)
        if float(score[i]) > 0:
            q,c,slot = map(int,item["actions"][i])
            admissible.append(dict(outcome,image=record["image"],query=q,proposed=c,slot=slot,
                                   score=float(score[i]),strict=strict,js=float(disagreement[i])))
        if record["image"] == 4793:
            matches = torch.where((item["actions"]==torch.tensor([139,0,60])).all(1))[0]
            if len(matches):
                j = int(matches[0])
                audit = dict(score=float(score[j]),rank=1+int((score>score[j]).sum()),
                             probabilities=average[j].tolist(),picked=j==i)
    return admissible,dict(first),audit


def evaluate(doc):
    receipt = read(OUT/"qualification.json")
    assert receipt["passed"] and receipt["plan"]==digest(PLAN) and receipt["index"]==digest(INDEX)
    assert receipt["encoder"]==digest(ENCODER) and receipt["checkpoint"]==digest(OUT/receipt["model"])
    model = Head().cuda().eval()
    model.load_state_dict(torch.load(OUT/receipt["model"],map_location="cpu",weights_only=True)["model"])
    policy_records,_ = collect(doc,"policy")
    policy_features, policy_counts = edge_records(doc,"policy")
    rows, policy_first, _ = rank(policy_records,policy_features,model)
    verified = [dict(row,correct=int(row["strict"]>=2)) for row in rows if row["strict"]>=0]
    cutoff = calibrate(verified)
    save_json(OUT/"policy.json",dict(cutoff=cutoff,first=policy_first,counts=policy_counts,
                                     qualification=digest(OUT/"qualification.json")))
    records,supply = collect(doc,"eval")
    eval_features, eval_counts = edge_records(doc,"eval")
    rows, first, audit = rank(records,eval_features,model)
    selected = [row for row in rows if passed(row["score"],cutoff["threshold"])]
    ap,classes = replay_metrics(doc,records,dict(retrained=selected))
    result = dict(images=supply["images"],classes=classes,first=first,counts=summarize(selected),
                  strict_correct=sum(r["strict"]>=2 for r in selected),strict_gain=sum(r["strict"]==3 for r in selected),
                  unknown=sum(r["target"]<0 for r in selected),gains=sum(state(r)==3 for r in selected),
                  cutoff=cutoff,eligible=len(rows),eval_counts=eval_counts,ap=ap,chair_to_person=audit,
                  frontier=frontier(rows,[],(0,1,2),True),selected=selected)
    save_json(OUT/"result.json",result)
    print(dict(result,selected=len(selected)),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase",choices=("train","repair","eval"))
    args = parser.parse_args()
    doc = read(INDEX)
    if args.phase == "eval":
        evaluate(doc)
    else:
        fit(doc,args.phase=="repair")
