"""Calibrate on separate images, then evaluate the locked selective detector."""
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
import torch

from selective import (AdoptionHead,FrozenStar,GAIN,apply_edits,count_edits,gt_arrays,
                       match,prepare_image,training_targets,utility,vocabulary,winners)
from evaluate import ap_interval,coco_eval,metrics
from train_selective import save_json


def fit_map(scores,targets):
    p=torch.tensor(scores,dtype=torch.float64).clamp(1e-5,1-1e-5)
    x=p.logit()
    y=torch.tensor(targets,dtype=torch.float64)
    if len(y)<30 or y.min()==y.max():
        return None
    start=torch.tensor((1-.05)/7.95,dtype=torch.float64).logit()
    weight=torch.tensor([float(start),0.],dtype=torch.float64,requires_grad=True)
    optimizer=torch.optim.LBFGS([weight],max_iter=80,line_search_fn="strong_wolfe")
    def params():
        return .05+7.95*weight[0].sigmoid(),20*weight[1].tanh()
    def closure():
        optimizer.zero_grad()
        a,b=params()
        loss=torch.nn.functional.binary_cross_entropy_with_logits(a*x+b,y)+1e-7*weight.square().sum()
        loss.backward()
        return loss
    optimizer.step(closure)
    a,b=params()
    assert torch.isfinite(a) and torch.isfinite(b)
    return [float(a.detach()),float(b.detach())]

def mapped_scores(probs,cal):
    if probs is None or cal["native"] is None or cal["edit"] is None:
        return None
    a,b=cal["native"]
    c,d=cal["edit"]
    logodds=torch.logit(probs[:,GAIN].clamp(1e-5,1-1e-5))
    return ((c*logodds+d-b)/a).sigmoid().clamp(max=cal["max_native"])


@torch.inference_mode()
def collect(manifest,split,star,head):
    names,nfine,parents=vocabulary(manifest,split)
    star.set_vocabulary(names,nfine)
    rows=[]
    for i,row in enumerate(manifest["splits"][split],1):
        base,prop,x=prepare_image(row,manifest,names,nfine,star,Path("cache")/split/(str(row["id"])+".pt"))
        probs=head(x.cuda(),prop["old"].cuda()).softmax(-1).cpu() if prop else None
        truth,y=gt_arrays(row,names,manifest)
        rows.append(dict(row=row,base=base,prop=prop,probs=probs,truth=truth,labels=y))
        if i%160==0 or i==len(manifest["splits"][split]):
            print(split,i,len(manifest["splits"][split]),flush=True)
    return rows,names,nfine


def score_maps(rows,manifest,names):
    source,source_y,target,target_y=[],[],[],[]
    for r in rows:
        base,prop,probs=r["base"],r["prop"],r["probs"]
        ids=match(base,r["truth"],r["labels"])
        source.extend(base["score"].tolist())
        source_y.extend((ids>=0).tolist())
        if prop:
            y,mask=training_targets(r["row"],manifest,names,base,prop)
            w=torch.tensor(winners(prop,probs))
            w=w[mask[w]]
            target.extend(probs[w,GAIN].tolist())
            target_y.extend((y[w]==GAIN).tolist())
    return dict(native=fit_map(source,source_y),edit=fit_map(target,target_y),
                max_native=max(source),native_pairs=len(source),edit_pairs=len(target),
                native_positives=sum(source_y),edit_positives=sum(target_y))


def evaluate_policy(rows,cutoff):
    stats=Counter()
    for r in rows:
        if cutoff is None or not r["prop"]:
            stats["old_true"]+=r["old_true"]
            continue
        result=apply_edits(r["base"],r["prop"],r["probs"],r["new_scores"],cutoff)
        if not (result["edit"]>=0).any():
            stats["old_true"]+=r["old_true"]
            continue
        d=count_edits(r["base"],result,r["truth"],r["labels"])
        stats.update(d)
        stats["images_with_edits"]+=1
    result=dict(stats)
    result["precision"]=stats["valid_edits"]/max(1,stats["edits"])
    result["negative_flip"]=stats["original_correct_replaced"]/max(1,stats["old_true"])
    return result


def calibrate(manifest,star,head):
    rows,names,_=collect(manifest,"cal_score",star,head)
    maps=score_maps(rows,manifest,names)
    print("maps",maps,flush=True)
    del rows
    rows,names,_=collect(manifest,"cal_policy",star,head)
    values=[]
    attainable=Counter()
    for r in rows:
        r["old_true"]=int((match(r["base"],r["truth"],r["labels"])>=0).sum())
        if r["prop"]:
            prop,probs=r["prop"],r["probs"]
            r["new_scores"]=mapped_scores(probs,maps)
            if r["new_scores"] is not None:
                w=winners(prop,probs)
                a=utility(probs)[w]
                values.extend(a[a>0].tolist())
            t,g=training_targets(r["row"],manifest,names,r["base"],prop)
            attainable.update(proposals=int(g.sum()),gain=int(((t==GAIN)&g).sum()))
            attainable["images_with_gain"]+=int(((t==GAIN)&g).any())
    thresholds=sorted({0.}|{float(x) for x in np.quantile(values,np.linspace(0,1,21))}) if values else []
    table=[]
    for cutoff in thresholds:
        c=evaluate_policy(rows,cutoff)
        good=c.get("valid_edits",0)>=10 and c["precision"]>=.90 and c["negative_flip"]<=.01
        table.append(dict(c,cutoff=cutoff,qualified=good))
    eligible=[r for r in table if r["qualified"]]
    best=max(eligible,key=lambda r:(r["new_instances"],-r.get("original_correct_replaced",0),r["precision"])) if eligible else None
    policy=dict(maps=maps,cutoff=None if best is None else best["cutoff"],selected=best,
                cal_policy_images=len(rows),candidate_ceiling=dict(attainable),thresholds=table)
    save_json("policy.json",policy)
    print("policy", {k:v for k,v in policy.items() if k!="thresholds"},flush=True)


def detection_list(row,out,catids):
    xywh=out["box"].clone()
    xywh[:,2:]-=xywh[:,:2]
    xywh*=torch.tensor([row["width"],row["height"],row["width"],row["height"]])
    return [dict(image_id=row["id"],category_id=catids[c],bbox=b,score=s)
            for c,b,s in zip(out["label"].tolist(),xywh.tolist(),out["score"].tolist())]


def precision_counts(ev,reference_tp=None,score_cut=.35):
    n=len(ev.params.imgIds)
    a=len(ev.params.areaRng)
    scores,hits=[],[]
    for k in range(len(ev.params.catIds)):
        for entry in ev.evalImgs[k*a*n:k*a*n+n]:
            if entry is None or not len(entry["dtScores"]):
                continue
            ok=entry["dtIgnore"][0]==0
            scores.extend(np.asarray(entry["dtScores"])[ok].tolist())
            hits.extend((entry["dtMatches"][0][ok]>0).tolist())
    scores,hits=np.asarray(scores),np.asarray(hits,dtype=bool)
    selected=scores>=score_cut
    raw=dict(tp=int((selected&hits).sum()),fp=int((selected&~hits).sum()))
    if reference_tp is not None:
        order=np.argsort(-scores,kind="stable")
        ordered=hits[order]
        cumulative=ordered.cumsum()
        pos=int(np.searchsorted(cumulative,reference_tp))
        raw["fp_at_reference_tp"]=int(pos+1-cumulative[pos]) if pos<len(ordered) else None
    return raw


def evaluate(manifest,star,head):
    policy=json.loads(Path("policy.json").read_text())
    rows,names,nfine=collect(manifest,"eval",star,head)
    catids=[manifest["categories"][c] for c in names[:nfine]]
    baseids={manifest["categories"][c] for c in manifest["base"]}
    result={}
    for group in ("random","enriched"):
        subset=[r for r in rows if (r["row"]["stratum"]=="random")== (group=="random")]
        baseline,new=[],[]
        stats=Counter()
        for r in subset:
            b,prop,probs=r["base"],r["prop"],r["probs"]
            scores=mapped_scores(probs,policy["maps"])
            out=apply_edits(b,prop,probs,scores,policy["cutoff"])
            edited=out["edit"]>=0
            # Every unedited detection retains box, class and native score exactly.
            u=out["origin"]>=0
            for k in b:
                assert torch.equal(out[k][u],b[k][out["origin"][u]])
            baseline.extend(detection_list(r["row"],b,catids))
            new.extend(detection_list(r["row"],out,catids))
            stats.update(count_edits(b,out,r["truth"],r["labels"]))
            stats["images_with_edits"]+=int(edited.any())
            if prop:
                t,m=training_targets(r["row"],manifest,names,b,prop)
                stats["eligible_proposals"]+=int(m.sum())
                stats["gain_proposals"]+=int(((t==GAIN)&m).sum())
                stats["images_with_gain_proposal"]+=int(((t==GAIN)&m).any())
        images=[r["row"] for r in subset]
        a=coco_eval(images,baseline,catids)
        b=coco_eval(images,new,catids) if baseline!=new else a
        ma,mb=metrics(a,baseids),metrics(b,baseids)
        ref=precision_counts(a)
        mb["confidence"]=precision_counts(b,ref["tp"])
        ma["confidence"]=ref
        result[group]=dict(images=len(images),dino=ma,selective=mb,edits=dict(stats),
                           ap_ci95=[0,0] if a is b else ap_interval(a,b))
        print(group,result[group],flush=True)
        save_json("result.json",result)


def main():
    if not json.loads(Path("qualification.json").read_text())["passed"]:
        raise SystemExit("Optimization qualification failed. No efficacy evaluation.")
    manifest=json.loads(Path("manifest.json").read_text())
    star=FrozenStar()
    state=torch.load("best.pt",map_location="cpu",weights_only=True)
    head=AdoptionHead(state["dim"]).cuda().eval()
    head.load_state_dict(state["state"])
    (calibrate if sys.argv[1]=="calibrate" else evaluate)(manifest,star,head)


if __name__=="__main__":
    main()
