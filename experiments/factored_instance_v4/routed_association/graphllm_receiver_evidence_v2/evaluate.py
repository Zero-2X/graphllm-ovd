"""Freeze policy with policy labels, then evaluate already sealed blind predictions."""
from collections import Counter
import csv
import importlib.metadata
from time import perf_counter
from pathlib import Path

import torch

from .common import OUT, annotated_split, cache_write, data_lock, digest, load, read, save_json
from .native_bridge import NativeBridge
from .predict import freeze_ids, key_index, one_slot, ready, score_rows, select
from .targets import STATUS, annotate, policy_prefix


@torch.inference_mode()
def policy_scores(cfg,shard):
    assert not (OUT/"locked_policy.json").exists()
    images=read(OUT/"images_policy.json")
    assert digest(OUT/"images_policy.json")==data_lock()["sets"]["policy"]["blind"]
    bridge=NativeBridge(images,cfg)
    model,gain,chain=ready(cfg)
    ci=key_index(bridge.categories)
    identity=freeze_ids(chain)
    provisional=dict(gain_logit_cutoff=None)
    population=images["rows"][shard::cfg["workers"]]
    lock=dict(chain=chain,blind=data_lock()["sets"]["policy"]["blind"])
    for start in range(0,len(population),cfg["block"]):
        frames=population[start:start+cfg["block"]]
        ids=[f["id"] for f in frames]
        p=OUT/f"policy_scores_{shard}_{start//cfg['block']:03}.pt"
        if p.exists():
            b=load(p)
            assert b["lock"]==lock and b["ids"]==ids
            continue
        scored=[]
        for frame in frames:
            row,scan=bridge.scan(frame)
            items=[(a,row["nodes"],row["local"]) for a in row["actions"]]
            records,_=score_rows(model,gain,bridge.risk,items,ci,cfg,identity)
            _,choice=select(records["full"],provisional)
            if choice:
                scored.append((choice["gain_logit"],choice["key"],row["base"],scan["pack"]["boxes"]))
        cache_write(p,dict(lock=lock,scored=scored,ids=ids))
        print("policy-scores",shard,start,len(scored),flush=True)


def lock_policy(cfg):
    from .common import model_lock

    path=OUT/"locked_policy.json"
    assert not path.exists()
    expected=read(OUT/"images_policy.json")
    chain=model_lock("policy")
    lock=dict(chain=chain,blind=data_lock()["sets"]["policy"]["blind"])
    seen=set()
    scored,hashes=[],{}
    for p in sorted(OUT.glob("policy_scores_*_*.pt")):
        b=load(p)
        assert b["lock"]==lock and not seen.intersection(b["ids"])
        seen.update(b["ids"])
        scored.extend(b["scored"])
        hashes[p.name]=digest(p)
    assert seen=={r["id"] for r in expected["rows"]}
    save_json(OUT/"policy_top1_manifest.json",dict(lock=lock,hashes=hashes,eligible=len(scored)))
    doc,rows=annotated_split("policy")
    by_frame={f["id"]:f for f in rows}
    values=[]
    for logit,key,base,boxes in scored:
        label=annotate(by_frame[key[0]],doc,dict(base=base),boxes,{key:[]})[key]
        values.append((logit,key,label))
    assert all(digest(OUT/name)==sha for name,sha in hashes.items())
    point=policy_prefix(values,cfg["policy"]["unknown_budget"])
    frozen=dict(chain=chain,mode="full",source=digest(OUT/"data_lock.json"),
                policy_top1=digest(OUT/"policy_top1_manifest.json"),
                eligible_images=len(values),policy_images=len(rows),**point)
    save_json(path,frozen)
    print("Policy",frozen,flush=True)


def observe(c,t):
    c.update(actions=1,confirmed_correct=int(t["confirmed_correct"]),original_class_harmed=int(t["original_class_harmed"]))
    c[t["state"]]+=1
    for j in range(6):
        c[f"new{j}"]+=t["new"][j]
        c[f"lost{j}"]+=t["lost"][j]


def first_stage(row,arm,good):
    valid=good.intersection(tuple(v["key"]) for v in row["records"][arm] if v["is_executable"])
    sem=valid.intersection(tuple(v["key"]) for v in row["records"][arm] if v["semantic_eligible"])
    top=sem.intersection(tuple(v["key"]) for v in row["records"][arm] if v["image_top1"])
    final=top.intersection(tuple(v["key"]) for v in row["records"][arm] if v["executed"])
    stages=(valid,sem,top,final)
    return [len(v) for v in stages]


def evaluate(cfg):
    from counterfactual import predictions
    from evaluate import coco_eval,metrics
    import pycocotools.cocoeval as coco

    assert not (OUT/"result.json").exists()
    seal=read(OUT/"predictions_manifest.json")
    assert seal["lock"]["ids"]["policy_hash"]==digest(OUT/"locked_policy.json")
    assert seal["lock"]["images"]==digest(OUT/"images_holdout.json")
    for name,sha in seal["hashes"].items():
        assert digest(OUT/name)==sha
    src=OUT/"manifest.json"
    assert digest(src)==data_lock()["annotations"]
    doc=read(src)
    rows=doc["rows"]
    ci=key_index(doc["categories"][s] for s in doc["base"])
    cat=list(ci)
    arms=cfg["evaluation"]["arms"]
    det={k:[] for k in arms}
    count={k:Counter() for k in arms}
    funnels={k:Counter() for k in ("full","off")}
    paired={k:Counter() for k in ("same","both_none","full_only","off_only","different")}
    pair_outcomes={k:{a:Counter() for a in ("v2_full","v2_message_off")} for k in paired}
    timing=Counter()
    opportunity=0
    total=0
    action_audit=[]
    opportunity_rows=[]
    migrations=Counter()
    by_id={r["id"]:r for r in rows}
    for name in seal["hashes"]:
        block=load(OUT/name)
        assert block["lock"]==seal["lock"]
        for row in block["rows"]:
            total+=1
            f=by_id[row["image"]]
            base=row["base"]
            boxes=torch.zeros((900,4)).index_copy(0,base["query"],base["box"])
            raw={tuple(k):[] for k in row["raw"]}
            targets=annotate(f,doc,row,boxes,raw,store_matches=True)
            good={k for k,t in targets.items() if t["gain_target"]==1}
            if good:
                opportunity+=1
                summary={}
                for name_ in ("full","off"):
                    counts=first_stage(row,name_,good)
                    summary[name_]=counts
                    funnels[name_].update({s:int(n>0) for s,n in zip(("valid","semantic","top","cutoff"),counts)})
                opportunity_rows.append(dict(image=f["id"],source_id=f["source_id"],opportunities=len(good),stages=summary))
            keys={k:tuple(v) if v is not None else None for k,v in row["keys"].items()}
            a,b=keys["v2_full"],keys["v2_message_off"]
            category=("both_none" if a==b is None else "same" if a==b else "full_only" if b is None
                      else "off_only" if a is None else "different")
            paired[category]["images"]+=1
            for arm,k in keys.items():
                changed=one_slot(base,k,ci)
                assert torch.equal(changed["label"],row["labels"][arm])
                det[arm].extend(predictions(f,changed,cat))
                if k is None:
                    continue
                t=targets[k]
                if arm=="frozen_v1_js_on":
                    migrations.update(correct_legacy=int(t["legacy_confirmed_correct"]),correct_v2=int(t["confirmed_correct"]),
                                      unknown_legacy=int(not t["legacy_utility_known"]),unknown_v2=int(not t["utility_known"]))
                observe(count[arm],t)
                if arm in pair_outcomes[category]:
                    observe(pair_outcomes[category][arm],t)
                action_audit.append(dict(arm=arm,key=k,source_id=f["source_id"],**t))
            timing.update(row["timing"])
            timing["actions"]+=len(row["records"]["full"])
    assert total==len(rows)==seal["images"]
    assert sum(c["images"] for c in paired.values())==total
    for suffix,arm in (("full","v2_full"),("off","v2_message_off")):
        assert funnels[suffix]["cutoff"]==count[arm]["safe_new"]
    annotations=[dict(a,image_id=f["id"],area=a["bbox"][2]*a["bbox"][3]) for f in rows for a in f["annotations"]]
    assert len({a["id"] for a in annotations})==len(annotations)
    result=[]
    start=perf_counter()
    baseline=None
    evaluator=None
    for arm in arms:
        assert len(det[arm])==len(det[arms[0]])
        ev=coco_eval(rows,det[arm],cat,annotations)
        params=ev.params
        evaluator=dict(code=digest(Path(coco.__file__)),
                       version=importlib.metadata.version("pycocotools"),ious=params.iouThrs.tolist(),
                       max_dets=params.maxDets,areas=params.areaRng,area_labels=params.areaRngLbl,
                       cats=[int(c) for c in params.catIds],
                       crowd="COCOeval bbox/default; separate replay uses project crowd test")
        values={k:v*100 for k,v in metrics(ev).items()}
        if baseline is None:
            baseline=values
        c=count[arm]
        assert sum(c[s] for s in STATUS)==c["actions"]
        assert not any(c[f"{key}{i}"] for key in ("new","lost") for i in (2,5))
        result.append(dict(arm=arm,**values,**{f"delta_{k}":v-baseline[k] for k,v in values.items()},
                      **{k:c[k] for k in ("actions",*STATUS,"confirmed_correct","original_class_harmed")},
                      **{f"{k}{i}":c[f"{k}{i}"] for k in ("new","lost") for i in range(6)},
                      opportunity_recall=c["safe_new"]/opportunity if opportunity else None))
        del ev
        print(arm,result[-1],flush=True)
    summary=dict(images=total,comparison="new project holdout",prediction_sha=digest(OUT/"predictions_manifest.json"),
                 rows=result,paired=paired,pair_outcomes=pair_outcomes,opportunity=opportunity,
                 funnels=funnels,migration=migrations,evaluator=evaluator,
                 timing_means={k:v/total for k,v in timing.items()},eval_seconds=perf_counter()-start)
    save_json(OUT/"action_audit.json",action_audit)
    save_json(OUT/"opportunity_images.json",opportunity_rows)
    with (OUT/"result.csv").open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(result[0]))
        writer.writeheader()
        writer.writerows(result)
    save_json(OUT/"result.json",summary)
