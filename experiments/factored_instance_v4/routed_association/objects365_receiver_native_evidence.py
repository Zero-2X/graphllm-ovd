"""Give the frozen GraphLLM verifier direct access to receiver class evidence."""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.nn.functional import softplus

from objects365_cache import block_path, split_rows
from objects365_coverage_cache import native_detector, native_evidence
from objects365_data import BLOCK, FOLDER, INDEX, SHARDS, digest, read, save_json
from objects365_donor_view_entropy import OUT as DONOR
from objects365_donor_view_entropy_eval import (BUDGETS, at_point, best, entropy, measure,
                                               paired_difference, prefix_curve, ranking)
from objects365_fit import replay_metrics
from objects365_receiver_pair_views import harmful
from objects365_receiver_token_probe import OUT as TOKEN

OUT = FOLDER/"receiver_native_evidence"
PLAN = Path(__file__).with_name("OBJECTS365_RECEIVER_NATIVE_EVIDENCE.md")
EPOCHS = 120


def anchors():
    return dict(index=digest(INDEX),plan=digest(PLAN),code=digest(Path(__file__)),
                diagnostic=digest(TOKEN/"result.json"),old=digest(DONOR/"result.json"))


def pairs(splits):
    evidence,ids = [],[]
    for split in splits:
        for path in sorted(TOKEN.glob(f"{split}_*.pt")):
            for iid,row in torch.load(path,map_location="cpu",weights_only=True)["records"].items():
                evidence.append(row["metrics"][:,:,[0,2]].double())
                ids.extend([iid]*len(row["donor"]))
    x = torch.cat(evidence)
    _,group = torch.unique(torch.tensor(ids),return_inverse=True)
    w = 1/torch.bincount(group)[group].double()
    w /= w.sum()
    return x,group,w


def split_values(raw, center, scale):
    return (raw[0][:,:,1].clamp(1e-7,1-1e-7).logit(), (raw[0][:,:,0]-center)/scale,*raw[1:])


def loss(alpha,data):
    p,m,_,_ = data
    combined = p+alpha*m
    gap = combined[:,0]-combined[:,1]
    return softplus(-gap),gap


@torch.no_grad()
def measure_pairs(alpha,data):
    _,_,i,w = data
    l,gap = loss(alpha,data)
    n = int(i.max())+1
    by_image = torch.zeros(n,dtype=l.dtype).scatter_add_(0,i,l*w)*n
    return dict(loss=float(by_image.mean()),correct=int((gap>0).sum()),pairs=len(gap)),by_image.numpy()


def fit():
    source = read(TOKEN/"result.json")
    assert source["max_score_gap"]<=.002
    train,monitor = pairs(("train","rare_train")),pairs(("monitor",))
    assert len(train[0])==source["train_pairs"] and len(monitor[0])==source["monitor_pairs"]
    center = float(train[0][:,:,0].mean())
    scale = float(train[0][:,:,0].std())
    assert scale>0 and np.isfinite(scale)
    train,monitor = split_values(train,center,scale),split_values(monitor,center,scale)
    alpha = torch.zeros((),dtype=torch.float64,requires_grad=True)
    initial, image_loss = measure_pairs(alpha,monitor)
    indexes = np.random.default_rng(20261112).integers(0,len(image_loss),(2000,len(image_loss)))
    lo,hi = np.quantile(image_loss[indexes].mean(1),(.025,.975))
    tol = float(max(.005,(hi-lo)/2))
    opt = torch.optim.SGD((alpha,),lr=.08)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS)
    best_result,best_weight,epoch = initial,0.,0
    history = []
    for e in range(1,EPOCHS+1):
        opt.zero_grad(set_to_none=True)
        l,_ = loss(alpha,train)
        value = l@train[-1]
        assert value.isfinite()
        value.backward()
        assert alpha.grad.isfinite()
        opt.step()
        schedule.step()
        train_value,_ = measure_pairs(alpha,train)
        valid,_ = measure_pairs(alpha,monitor)
        weight = float(alpha.detach())
        history.append(dict(epoch=e,train=train_value,monitor=valid,weight=weight,lr=schedule.get_last_lr()[0]))
        if valid["loss"]<best_result["loss"]:
            best_result,best_weight,epoch = valid,weight,e
    end = np.array([r["monitor"]["loss"] for r in history[-5:]])
    tr = [r["train"]["loss"] for r in history[-5:]]
    slope = float(np.polyfit(np.arange(5),end,1)[0])
    checks = dict(schedule=history[-1]["lr"]==0,plateau=abs(tr[-1]-tr[0])/tr[0]<=.025,
                  slope=abs(slope)<=tol/4,spread=bool(np.ptp(end)<=tol),gap=bool(end[-1]-best_result["loss"]<=tol))
    receipt = dict(anchor=anchors(),checks=checks,passed=all(checks.values()),initial=initial,best=best_result,
                   best_epoch=epoch,best_weight=best_weight,final=history[-1],center=center,scale=scale,tolerance=tol)
    save_json(OUT/"history.json",history)
    save_json(OUT/"qualification.json",receipt)
    print(receipt,flush=True)


def qualified():
    receipt = read(OUT/"qualification.json")
    assert receipt["anchor"]==anchors() and receipt["passed"] and receipt["best_epoch"]>0
    return receipt


def blocks(doc,split,shard):
    frames = split_rows(doc,split,shard)
    for start in range(0,len(frames),BLOCK):
        subset = frames[start:start+BLOCK]
        part = start//BLOCK
        name = f"{split}_{shard}_{part:03d}.pt"
        yield subset,block_path(split,shard,part),DONOR/name,OUT/name


@torch.inference_mode()
def extract(doc,split,shard,receipt):
    if split=="eval":
        policy = read(OUT/"policy.json")
        assert policy["qualification"]==digest(OUT/"qualification.json")
    detector = native_detector(doc)
    for frames,source,view,path in blocks(doc,split,shard):
        lock = dict(anchor=receipt["anchor"],source=digest(source),view=digest(view),images=[r["id"] for r in frames])
        if path.exists():
            assert torch.load(path,map_location="cpu",weights_only=True)["lock"]==lock
            continue
        old = torch.load(source,map_location="cpu",weights_only=True)
        graph = torch.load(view,map_location="cpu",weights_only=True)
        assert graph["lock"]["source"]==lock["source"] and graph["lock"]["images"]==lock["images"]
        frame_by_id = {r["id"]:r for r in frames}
        records = {}
        for row in old["records"]:
            if "actions" not in row:
                continue
            iid = row["image"]
            before = graph["records"][iid]
            pack,base = native_evidence(detector,frame_by_id[iid])
            q,c,slot = row["actions"].T
            assert torch.equal(base["query"][slot],q) and torch.equal(base["label"][slot],before["old"])
            assert torch.equal(row["actions"],before["actions"]) and torch.equal(base["score"][slot],before["score"])
            if "base" in row:
                assert all(torch.equal(base[k],row["base"][k]) for k in base) and torch.equal(pack["boxes"],row["boxes"])
            margin = pack["logits"][q,c]-pack["logits"][q,before["old"]]
            assert margin.isfinite().all()
            records[iid] = margin
        assert records.keys()==graph["records"].keys()
        temp = path.with_suffix(".part")
        torch.save(dict(lock=lock,records=records),temp)
        temp.replace(path)
        print(path.name,len(records),flush=True)


def scores(doc,split,receipt):
    first,multi,native,trace = [],[],[],{}
    counts = Counter()
    for shard in range(SHARDS):
        for frames,source,view,path in blocks(doc,split,shard):
            block = torch.load(path,map_location="cpu",weights_only=True)
            expected = dict(anchor=receipt["anchor"],source=digest(source),view=digest(view),images=[r["id"] for r in frames])
            assert block["lock"]==expected
            graph = torch.load(view,map_location="cpu",weights_only=True)["records"]
            records = torch.load(source,map_location="cpu",weights_only=True)["records"]
            native.extend(records)
            counts["images"] += len(frames)
            for row in records:
                if "actions" not in row:
                    continue
                iid = row["image"]
                old = graph[iid]
                p = old["views"].double().clamp(1e-7,1-1e-7)
                margin = block["records"][iid].double()
                calibrated = (p.logit()+receipt["best_weight"]*((margin-receipt["center"])/receipt["scale"])[:,None]).sigmoid()
                mean = calibrated.mean(1)
                js = (entropy(mean)-entropy(calibrated).mean(1)).clamp_min(0)
                pick = int(mean.argmax())
                two = set(old["selected"].tolist())
                counts.update(candidate_images=1,possible=int(any(x["correct"] for x in row["outcomes"])),
                              correct_first=row["outcomes"][pick]["correct"],multi_first=int(pick in two))
                for j in two|{pick}:
                    q,c,s = row["actions"][j].tolist()
                    y = row["outcomes"][j]
                    x = dict(y,image=iid,query=q,proposed=c,slot=s,score=float(mean[j]),
                             js=float(js[j]) if j in two else None,harmful=bool(harmful(y)))
                    if j==pick:
                        first.append(x)
                    if j in two:
                        multi.append(x)
                if iid==4793:
                    for j,(q,c,s) in enumerate(row["actions"].tolist()):
                        if (q,c)==(139,0):
                            trace[str(iid)] = dict(score=float(mean[j]),rank=int((mean>mean[j]).sum())+1,
                                                   js=float(js[j]) if j in two else None,margin=float(margin[j]),
                                                   old_score=float(p[j].mean()))
    return first,multi,native,dict(counts),trace


def policy(doc,receipt):
    first,_,_,counts,_ = scores(doc,"policy",receipt)
    data = [r["js"] for r in first if r["js"] is not None and r["correct"] and r["score"]>.50]
    cap = float(np.quantile(data,.90)) if data else None
    points = {}
    for name,limit in (("mean",None),("mixed_entropy",cap)):
        curve = prefix_curve(first,limit,counts)
        points[name] = {str(b):best(curve,b) for b in BUDGETS}
    result = dict(qualification=digest(OUT/"qualification.json"),cap=cap,counts=counts,points=points)
    save_json(OUT/"policy.json",result)
    print("Policy",result,flush=True)


def evaluate(doc,receipt):
    settings = read(OUT/"policy.json")
    assert settings["qualification"]==digest(OUT/"qualification.json")
    first,multi,native,counts,trace = scores(doc,"eval",receipt)
    caps = {"mean":None,"mixed_entropy":settings["cap"]}
    accepted = {name:{b:at_point(first,settings["points"][name][str(b)]["cutoff"],limit) for b in BUDGETS}
                for name,limit in caps.items()}
    actual = {name:{b:measure(v,counts["possible"],len(first)) for b,v in choices.items()}
              for name,choices in accepted.items()}
    curves = {name:prefix_curve(first,limit,counts) for name,limit in caps.items()}
    ap,classes = replay_metrics(doc,native,{name:rows[0] for name,rows in accepted.items()})
    result = dict(qualification=settings["qualification"],policy=digest(OUT/"policy.json"),
                  counts=counts,actual=actual,trace=trace,accepted=accepted,
                  frontier={name:{b:best(curve,b) for b in BUDGETS} for name,curve in curves.items()},
                  multi_ranking=ranking(multi),ap=ap,classes=classes,paired=paired_difference(doc,accepted))
    save_json(OUT/"result.json",result)
    print("Result",counts,actual,trace,result["multi_ranking"],ap,flush=True)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage",choices=("fit","extract","policy","eval"))
    parser.add_argument("--split",choices=("policy","eval"))
    parser.add_argument("--shard",type=int,choices=range(SHARDS))
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.stage=="fit":
        fit()
    else:
        doc = read(INDEX)
        receipt = qualified()
        if args.stage=="extract":
            extract(doc,args.split,args.shard,receipt)
        else:
            (policy if args.stage=="policy" else evaluate)(doc,receipt)

