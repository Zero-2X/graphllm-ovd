"""Lock one operating point before reading efficacy labels."""
from collections import Counter
from itertools import groupby
import json
from pathlib import Path
import sys

import numpy as np
import torch

from counterfactual import ImageReplay,make_base,predictions,save_json,vocabulary
from evaluate import ap_interval,coco_eval,metrics
from finish import precision_counts
from gate import FrozenEvidence,read_model,select_action


def describe(replay,action,score):
    new,lost,fp,sem,target=replay.delta(action,score)
    gain=target>=0 and (any(new) or min(fp)<0 or sem>0)
    harm=any(lost) or max(fp)>0 or sem<0
    return dict(valid=int(target>=0),gain=int(gain),strict_gain=int(gain and not harm),harm=int(harm),
                semantic_fix=int(sem>0),semantic_bad=int(sem<0),new_floor=len(new[0]),
                lost_floor=len(lost[0]),new_035=len(new[2]),lost_035=len(lost[2]),fp_delta_035=fp[2])


def denominators(replay,actions):
    old_true=sum(len(c[0][0]) for c in replay.before.values())
    if not replay.iou.shape[1] or not actions:
        return old_true,0
    own={}
    for q,c in zip(replay.bq,replay.bc):
        own.setdefault(int(q),int(c))
    correct=0
    for q in {a[1] for a,_ in actions} & set(own):
        k=int(replay.iou[q].argmax())
        correct+=int(replay.iou[q,k]>=.50 and replay.y[k]==own[q])
    return old_true,correct


@torch.inference_mode()
def collect(manifest,split,reader,model):
    names,nfine,parents=vocabulary(manifest,split)
    reader.vocabulary(names,nfine,parents)
    entries=[]
    for i,row in enumerate(manifest["splits"][split],1):
        pack=torch.load(Path("cache")/split/(str(row["id"])+".pt"),map_location="cpu",weights_only=True)
        base=make_base(pack,nfine)
        actions,x=reader.encode(pack,base)
        chosen=select_action(actions,x,base,*model)
        replay=ImageReplay(row,pack,base,manifest,names)
        n_true,n_correct=denominators(replay,actions)
        item=dict(row=row,base=base,old_true=n_true,correct_receivers=n_correct,possible_actions=len(actions))
        if chosen is not None:
            ix,utility,sem=chosen
            action,score=actions[ix]
            item.update(action=action,score=score,utility=utility,semantic_estimates=sem,
                        outcome=describe(replay,action,score),modified=replay.output(action,score))
        entries.append(item)
        if i%160==0 or i==len(manifest["splits"][split]):
            print(split,i,len(manifest["splits"][split]),flush=True)
    return entries,names,nfine


def qualified(stats,old_true,correct_receivers):
    edits=stats["edits"]
    if not edits:
        return False
    extra_fp=stats["fp_delta_035"]
    extra_tp=stats["new_035"]-stats["lost_035"]
    return (stats["strict_gain"]>=5 and stats["valid"]/edits>=.80 and
            stats["semantic_bad"]<=.01*correct_receivers and stats["lost_floor"]<=.01*old_true and
            extra_fp<=.10*max(0,extra_tp))


def summarize(entries,cutoff):
    out=Counter()
    for row in entries:
        out["old_true"]+=row["old_true"]
        out["correct_receivers"]+=row["correct_receivers"]
        out["possible_actions"]+=row["possible_actions"]
        out["images"]+=1
        if cutoff is not None and row.get("utility",-float("inf"))>=cutoff:
            out.update(row["outcome"])
            out["edits"]+=1
            out[row["action"][0]+"_edits"]+=1
            out[row["action"][0]+"_strict_gain"]+=row["outcome"]["strict_gain"]
            out[row["action"][0]+"_semantic_fix"]+=row["outcome"]["semantic_fix"]
    return dict(out)


def calibrate(manifest,reader,model):
    entries,_,_=collect(manifest,"cal_policy",reader,model)
    old_true=sum(x["old_true"] for x in entries)
    correct_receivers=sum(x["correct_receivers"] for x in entries)
    order=sorted((x for x in entries if "utility" in x),key=lambda r:-r["utility"])
    counter=Counter()
    best=None
    best_key=None
    history=[]
    for cutoff,rows in groupby(order,key=lambda r:r["utility"]):
        for row in rows:
            counter.update(row["outcome"])
            counter["edits"]+=1
        is_qualified=qualified(counter,old_true,correct_receivers)
        history.append(dict(counter,cutoff=cutoff,qualified=is_qualified))
        if is_qualified:
            key=(counter["semantic_fix"]-counter["semantic_bad"],counter["strict_gain"],
                 counter["new_035"]-counter["lost_035"])
            if best is None or key>best_key:
                best=history[-1]
                best_key=key
    cutoff=None if best is None else best["cutoff"]
    result=dict(cutoff=cutoff,selected=summarize(entries,cutoff),
                unfiltered_winners=summarize(entries,order[-1]["utility"] if order else None),
                thresholds=history)
    save_json("policy.json",result)
    print("policy",{k:v for k,v in result.items() if k!="thresholds"},flush=True)


def evaluate(manifest,reader,model):
    cutoff=json.loads(Path("policy.json").read_text())["cutoff"]
    entries,names,nfine=collect(manifest,"eval",reader,model)
    catids=[manifest["categories"][c] for c in names[:nfine]]
    baseids={manifest["categories"][c] for c in manifest["base"]}
    report={}
    for group in ("random","enriched"):
        subset=[r for r in entries if (r["row"]["stratum"]=="random")== (group=="random")]
        raw=[]
        gated=[]
        for item in subset:
            base=item["base"]
            modified=item["modified"] if cutoff is not None and item.get("utility",-float("inf"))>=cutoff else base
            raw.extend(predictions(item["row"],base,catids))
            gated.extend(predictions(item["row"],modified,catids))
        rows=[r["row"] for r in subset]
        baseline=coco_eval(rows,raw,catids)
        new=baseline if gated==raw else coco_eval(rows,gated,catids)
        target=precision_counts(baseline)["tp"]
        dino=metrics(baseline,baseids)
        graph=metrics(new,baseids)
        dino["working_point"]=precision_counts(baseline,target)
        graph["working_point"]=precision_counts(new,target)
        stats=summarize(subset,cutoff)
        samples=[]
        for r in subset:
            if cutoff is not None and r.get("utility",-float("inf"))>=cutoff:
                samples.append(dict(image_id=r["row"]["id"],action=r["action"][0],query=r["action"][1],
                                    label=names[r["action"][2]],score=r["score"],utility=r["utility"],
                                    outcome=r["outcome"],sem=r["semantic_estimates"]))
        report[group]=dict(dino=dino,graph=graph,stats=stats,
                           ap_ci95=[0.,0.] if baseline is new else ap_interval(baseline,new),
                           actions=samples)
        save_json("result.json",report)
        print(group,{k:v for k,v in report[group].items() if k!="actions"},flush=True)


if __name__=="__main__":
    mode=sys.argv[1]
    assert mode in {"calibrate","eval"}
    if not json.loads(Path("qualification.json").read_text())["passed"]:
        raise SystemExit("Optimization not qualified. Policy and efficacy unopened.")
    if mode=="eval" and json.loads(Path("policy.json").read_text())["cutoff"] is None:
        save_json("eval_deferred.json",dict(opened=False,reason="No qualified nonempty policy on independent calibration."))
        print("Eval remains sealed because calibration selected keep-all.",flush=True)
        raise SystemExit(0)
    manifest=json.loads(Path("manifest.json").read_text())
    model=read_model()
    reader=FrozenEvidence()
    (calibrate if mode=="calibrate" else evaluate)(manifest,reader,model)
