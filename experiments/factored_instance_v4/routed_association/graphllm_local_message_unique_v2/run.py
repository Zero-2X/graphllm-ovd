"""Build, train once, freeze, predict blind, and evaluate once."""
import argparse
from collections import Counter
import csv
import os
from pathlib import Path
from time import perf_counter

import torch
import torch.distributed as dist

from .bridge import (BASE, CODE, CONFIG, FOLDER, INDEX, KEEP, OUT, SHARDS, Vision, cache_write, data_sources, digest,
                     fresh_frame_digest, load, prepare_resolved, read, safe_frame, save_json, target_table)
from .core import SCHEMA, apply, choose_prefix, fuse, gain_features, pack_actions, select
from .train import base_lock, fitted_gain, fitted_reader, records, train_gain, train_semantic


def frozen_config(path):
    current = read(OUT/"resolved_config.json")
    assert current["config_hash"]==digest(path) and current["schema"]==SCHEMA
    for name,h in current["code_hashes"].items():
        assert digest(CODE/name)==h,name
    for name,stamp in current["source_stamps"].items():
        path=Path(name)
        stat=path.stat()
        if [stat.st_size,stat.st_mtime_ns]!=stamp:
            assert digest(path)==current["source_hashes"][name],f"Changed source artifact: {name}"
    return current


def require_build():
    stage = read(OUT/"build_receipt.json")
    assert stage["complete"] and stage["resolved"]==base_lock()
    assert (OUT/"BUILD_COMPLETE.md").is_file()
    assert digest(OUT/"native5_train.pt")==stage["normalizer"]
    return stage


def seal():
    from .train import groups, weighted_epoch
    check = read(OUT/"smoke.json")
    source = FOLDER/"graphllm_local_message_v1"
    old = read(source/"build_receipt.json")
    old_lock = digest(source/"resolved_config.json")
    assert check["resolved"]==base_lock() and all(check["checks"].values())
    assert old["resolved"]==old_lock==read(OUT/"resolved_config.json")["visual_source"]
    assert len(old["cache_hashes"])==256
    for name, sha in old["cache_hashes"].items():
        assert digest(source/name)==sha,name
    path = source/"native5_train.pt"
    assert digest(path)==old["normalizer"]
    norm = load(path)
    assert norm["lock"]==old_lock
    cache_write(OUT/"native5_train.pt",dict(norm,lock=base_lock()))
    receipt = dict(old,resolved=base_lock(),visual_source=old_lock,checks=check["checks"],
                   normalizer=digest(OUT/"native5_train.pt"))
    save_json(OUT/"build_receipt.json",receipt)
    train = [r for r in records("train") if r["semantic_target"]>=0]
    table,w = groups(train,"semantic_target",3)
    order = weighted_epoch(len(train),read(OUT/"resolved_config.json")["config"]["seed"]+1)
    assert sorted(order)==list(range(len(train)))
    assert all(abs(float(w[[i for ids in g.values() for i in ids]].sum())-1/3)<1e-10
               for g in table.values())
    assert all(abs(float(w[ids].sum())-1/(3*len(g)))<1e-10
               for g in table.values() for ids in g.values())
    # Match the gradient scaling of the actual loop, including the last partial batch.
    for count in (64,len(train)%64 or 64):
        padded = order[:count]+[order[0]]*(64-count)
        accum=[]
        for rank in range(4):
            values=torch.zeros(64,dtype=torch.float64)
            for pos in (rank*8,32+rank*8):
                end=min(pos+8,count)
                if end>pos:
                    values[pos:end] = w[padded[pos:end]]*4*len(train)/count
            accum.append(values)
        expect=torch.zeros(64,dtype=torch.float64)
        expect[:count]=w[padded[:count]]*len(train)/count
        torch.testing.assert_close(torch.stack(accum).mean(0),expect,rtol=0,atol=0)
    (OUT/"BUILD_COMPLETE.md").write_text(
        "# BUILD_COMPLETE\n\n"
        "原生视觉缓存与标签沿用已封存的 v1 数据，256 个视觉块的哈希全部核对。"
        "本轮八组检查、训练采样无重复、类别与类内图像质量平衡、"
        "四卡梯度权重和末批次缩放检查全部通过。\n",encoding="utf-8")
    print("BUILD_COMPLETE",len(train),flush=True)


@torch.no_grad()
def decide(keys, z, donor, rho, native16, risk, model, cfg, lam, cutoff, metadata=None):
    fusion=fuse(z,donor,rho,cfg["message"],False,lam)
    zz=fusion["fused"]
    score=model(gain_features(native16,zz)).sigmoid()
    logs=[]
    for i,key in enumerate(keys):
        finite=all(t[i].isfinite().all() for t in (z,donor,rho,native16,risk,zz,score))
        valid=bool(fusion["has_valid_donor"][i])
        semantic=bool(finite and zz[i,1]>max(zz[i,0],zz[i,2]))
        reason=None
        if not finite:
            reason="invalid_input"
        elif not valid:
            reason="no_valid_donor"
        elif not semantic:
            reason="semantic_not_c1"
        elif float(risk[i])>cfg["risk_cap"]:
            reason="risk_reject"
        def numerical(value):
            return value[i].tolist() if finite else None
        logs.append(dict(key=key,z_self=numerical(z),z_donor=numerical(donor),rho=numerical(rho),
                         z_fused=numerical(zz),delta=numerical(fusion["delta"]),js=numerical(fusion["js"]),
                         weights=numerical(fusion["weights"]),js_available=bool(fusion["js_available"][i]),
                         quality_abs=numerical(fusion["quality_abs"]),beta=numerical(fusion["beta"]),
                         p_other=numerical(zz.softmax(-1)[:,2]),risk_score=numerical(risk),gain_score=numerical(score),
                         is_executable=finite and valid,semantic_eligible=semantic,selected=False,executed=False,
                         reject_reason=reason,**(metadata[i] if metadata else {})))
    winner=select(logs)
    for r in logs:
        if r is winner:
            r["selected"]=True
            r["executed"]=cutoff is not None and r["gain_score"]>=cutoff
            if not r["executed"]:
                r["reject_reason"]="below_gain_cutoff"
        elif r["reject_reason"] is None:
            r["reject_reason"]="not_image_top1"
    return winner,logs


def lock_policy(cfg):
    require_build()
    assert not (OUT/"locked_policy.json").exists(),"Policy already locked"
    for f in ("semantic","gain"):
        assert read(OUT/f"{f}_qualification.json")["passed"],f"{f} failed optimization qualification"
    gain=fitted_gain()
    from objects365_receiver_keep_gate import qualified
    risk,_=qualified()
    cache=load(OUT/"semantics_policy.pt")
    assert cache["lock"]==base_lock() and cache["semantic"]==digest(OUT/"semantic_best.pt")
    from collections import defaultdict
    groups=defaultdict(list)
    for row in cache["rows"]:
        groups[row["key"][0]].append(row)
    top=[]
    for rows in groups.values():
        n=len(rows)
        x=torch.stack([r["features"][:16] for r in rows])
        z=torch.stack([r["z_self"] for r in rows])
        d=torch.stack([r["z_donor"] for r in rows])
        rho=torch.zeros(n,2)
        for i,r in enumerate(rows):
            rho[i,:len(r["rho"])]=r["rho"]
        chosen,_=decide([r["key"] for r in rows],z,d,rho,x,risk(x).sigmoid(),gain,cfg,cfg["message"]["lambda"],None)
        if chosen is not None:
            y=next(r["utility"] for r in rows if r["key"]==chosen["key"])
            top.append(dict(gain_score=chosen["gain_score"],target=y))
    point=choose_prefix(top,cfg["unknown_budget"])
    frozen=dict(schema=SCHEMA,resolved=base_lock(),semantic=cache["semantic"],gain=digest(OUT/"gain_best.pt"),
                risk=digest(KEEP/"best.pt"),risk_cap=cfg["risk_cap"],**point)
    save_json(OUT/"locked_policy.json",frozen)
    print(frozen,flush=True)


@torch.inference_mode()
def predict(cfg,images,shard):
    require_build()
    policy=read(OUT/"locked_policy.json")
    assert policy["resolved"]==base_lock() and policy["schema"]==SCHEMA
    assert policy["semantic"]==digest(OUT/"semantic_best.pt") and policy["gain"]==digest(OUT/"gain_best.pt")
    assert policy["risk"]==digest(KEEP/"best.pt")
    manifest=read(images)
    assert set(manifest)=={"names","base","parents","rows"}
    assert fresh_frame_digest(manifest)==read(OUT/"resolved_config.json")["blind_hash"]
    assert all(set(f)=={"id","source_id","file_name","path","width","height","sha"} for f in manifest["rows"])
    lock=dict(policy=digest(OUT/"locked_policy.json"),images=digest(images),schema=SCHEMA,
              semantic=policy["semantic"],gain=policy["gain"])
    vision=Vision(manifest,legacy=True)
    model=fitted_reader(manifest,cfg)
    gain=fitted_gain()
    rows=manifest["rows"][shard::SHARDS]
    for offset in range(0,len(rows),32):
        batch=rows[offset:offset+32]
        path=OUT/f"prediction_{shard}_{offset//32:03}.pt"
        ids=[r["id"] for r in batch]
        if path.exists():
            current=load(path)
            assert current["lock"]==lock and current["ids"]==ids
            continue
        output=[]
        for frame in batch:
            scan=vision.scan(frame)
            start=perf_counter()
            legacy=vision.legacy_pick(frame,scan)
            torch.cuda.synchronize()
            legacy_time=perf_counter()-start
            selected=dict(native=None,legacy_locked=legacy,new_full=None,new_js_off=None)
            actions=scan["actions"]
            logs={}
            qwen_time=0.
            decision_times={}
            if actions:
                start=perf_counter()
                all_z,all_d,all_r,all_x=[],[],[],[]
                for i in range(0,len(actions),32):
                    packed=pack_actions(actions[i:i+32],"cuda")
                    with torch.autocast("cuda",dtype=torch.bfloat16,enabled=torch.cuda.is_bf16_supported()):
                        z,d=model(packed)
                    all_z.append(z.cpu())
                    all_d.append(d.cpu())
                    all_r.append(packed["rho"].cpu())
                    all_x.append(packed["native16"].cpu())
                torch.cuda.synchronize()
                qwen_time=perf_counter()-start
                z,d,rho,x=[torch.cat(v) for v in (all_z,all_d,all_r,all_x)]
                risk=vision.risk(x).sigmoid()
                meta=[{k:a[k] for k in ("donor_query_ids","donor_predicted_classes")} for a in actions]
                for name,lam in (("new_full",cfg["message"]["lambda"]),("new_js_off",0.)):
                    start=perf_counter()
                    winner,log=decide([a["key"] for a in actions],z,d,rho,x,risk,gain,cfg,lam,policy["gain_cutoff"],meta)
                    if winner and winner["executed"]:
                        selected[name]=winner["key"]
                    logs[name]=log
                    decision_times[name]=perf_counter()-start
            base=scan["base"]
            labels={arm:apply(base,key,reject_duplicate=arm!="legacy_locked")["label"] for arm,key in selected.items()}
            oracle=[(frame["id"],q,s,int(base["label"][s]),c) for q,c,s in scan["raw"].tolist()]
            output.append(dict(image=frame["id"],base=base,selected=selected,labels=labels,actions=oracle,
                               records=logs,rejected=scan["rejected"],timing=dict(native=scan["native_time"],
                               visual=scan["feature_time"],legacy=legacy_time,qwen=qwen_time,decision=decision_times)))
        cache_write(path,dict(lock=lock,ids=ids,rows=output))
        print("predict",shard,offset,len(output),flush=True)


def seal_predictions():
    images=OUT/"holdout_images.json"
    expected={r["id"] for r in read(images)["rows"]}
    paths=sorted(OUT.glob("prediction_*.pt"))
    seen=set()
    files={}
    for path in paths:
        block=load(path)
        lock=block["lock"]
        assert lock["policy"]==digest(OUT/"locked_policy.json") and lock["images"]==digest(images)
        assert block["ids"]==[r["image"] for r in block["rows"]] and not seen.intersection(block["ids"])
        seen.update(block["ids"])
        files[path.name]=digest(path)
    assert seen==expected
    save_json(OUT/"predictions_manifest.json",dict(policy=digest(OUT/"locked_policy.json"),images=digest(images),
                                                   files=files,images_count=len(seen)))


def evaluate():
    import numpy as np
    from counterfactual import predictions
    from evaluate import coco_eval,metrics
    from omni_layer_cache import adjudicate
    if (OUT/"result.json").exists():
        raise RuntimeError("The one holdout comparison is already terminal")
    frozen=read(OUT/"predictions_manifest.json")
    assert frozen["policy"]==digest(OUT/"locked_policy.json")
    assert frozen["images"]==digest(OUT/"holdout_images.json")
    for n,h in frozen["files"].items():
        assert digest(OUT/n)==h
    manifest=read(OUT/"manifest.json")
    resolved=read(OUT/"resolved_config.json")
    path=OUT/"manifest.json"
    assert digest(path)==resolved["source_hashes"][str(path.resolve())]
    frames=manifest["rows"]
    cat_ids=[manifest["categories"][s] for s in manifest["base"]]
    arms=resolved["config"]["arms"]
    preds={arm:[] for arm in arms}
    counts={arm:Counter() for arm in arms}
    costs={arm:[] for arm in arms}
    flow=Counter()
    flow_values={k:[] for k in ("delta_inf","js","beta")}
    n=0
    opportunities=0
    for filename in frozen["files"]:
        for r in load(OUT/filename)["rows"]:
            n+=1
            f=frames[r["image"]]
            base=r["base"]
            raw_keys=r["actions"]
            targets={}
            if raw_keys:
                actions=torch.tensor([(k[1],k[4],k[2]) for k in raw_keys])
                observed=adjudicate(f,dict(boxes=torch.zeros((900,4)).index_copy(0,base["query"],base["box"])),
                                    base,manifest,actions)
                targets={tuple(k):v for k,v in zip(raw_keys,observed)}
            from .core import gain_label
            opportunities+=any(gain_label(v)==1 for v in targets.values())
            for arm in arms:
                key=r["selected"][arm]
                out=apply(base,key,reject_duplicate=arm!="legacy_locked")
                assert torch.equal(r["labels"][arm],out["label"])
                protected=base["score"]>=.35
                assert torch.equal(out["label"][protected],base["label"][protected])
                if arm=="legacy_locked" and key is not None:
                    slot=key[2]
                    order=torch.cat((torch.arange(len(base["label"]))[torch.arange(len(base["label"]))!=slot],
                                     torch.tensor([slot])))
                    out={k:v[order] for k,v in out.items()}
                preds[arm].extend(predictions(f,out,cat_ids))
                t=r["timing"]
                costs[arm].append(0. if arm=="native" else t["visual"]+
                    (t["legacy"] if arm=="legacy_locked" else t["qwen"]+t["decision"].get(arm,0.)))
                if key is None:
                    continue
                target=targets[tuple(key)]
                c=counts[arm]
                c.update(actions=1,confirmed_correct=target["correct"],known_harm=int(bool(target["wrong"] or any(target["lost"]))),
                         unconfirmed=int(target["target"]<0),lossless_gain_actions=int(gain_label(target)==1),
                         original_match_losses=sum(target["lost"]),loss_actions=int(any(target["lost"])))
                for i in range(6):
                    c[f"new_{i}"]+=target["new"][i]
                    c[f"lost_{i}"]+=target["lost"][i]
            for rec in r["records"].get("new_full",[]):
                flow.update(candidates=1,two_donors=int(rec["js_available"]),semantic=int(rec["semantic_eligible"]),
                            risk_pass=int(rec["semantic_eligible"] and rec["risk_score"]<=resolved["config"]["risk_cap"]),
                            executed=int(rec["executed"]))
                if rec["is_executable"]:
                    flow_values["delta_inf"].append(max(abs(v) for d in rec["delta"] for v in d))
                    flow_values["js"].append(rec["js"])
                    flow_values["beta"].append(rec["beta"])
    assert n==len(frames)==frozen["images_count"]
    annotations=[dict(a,image_id=f["id"],area=a["bbox"][2]*a["bbox"][3]) for f in frames for a in f["annotations"]]
    for i,a in enumerate(annotations,1):
        a["id"]=i
    output=[]
    for arm in arms:
        c=counts[arm]
        ap=metrics(coco_eval(frames,preds[arm],cat_ids,annotations))
        row=dict(arm=arm,**ap,actions=c["actions"],coverage=c["actions"]/n,confirmed_correct=c["confirmed_correct"],
                 lossless_gain_actions=c["lossless_gain_actions"],lossless_gain_opportunity_recall=
                 c["lossless_gain_actions"]/opportunities if opportunities else None,known_harm=c["known_harm"],
                 original_match_losses=c["original_match_losses"],unconfirmed=c["unconfirmed"],
                 new50=c["new_0"],lost50=c["lost_0"],net50=c["new_0"]-c["lost_0"],
                 new75=c["new_3"],lost75=c["lost_3"],net75=c["new_3"]-c["lost_3"],
                 extra_ms=1000*float(np.mean(costs[arm])))
        output.append(row)
    with (OUT/"result.csv").open("w",encoding="utf-8",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    medians={k:float(np.median(v)) if v else None for k,v in flow_values.items()}
    result=dict(predictions=digest(OUT/"predictions_manifest.json"),images=n,opportunities=opportunities,rows=output,
                flow=dict(flow),medians=medians,six_level={arm:dict(counts[arm]) for arm in arms},
                timing="One actual DINO forward per image, shared exact visual features. Real Qwen forward included. D shares C Qwen time and uses its own measured fusion.")
    save_json(OUT/"result.json",result)
    lines=["# RESULT","","本次完成同一项目留出清单上的一次四臂比较。","","| 臂 | AP | AP50 | AP75 | 放行 | 改对 | 有害 | 未确证 | 净匹配 0.50 / 0.75 |","|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in output:
        lines.append(f"| {r['arm']} | {r['AP']*100:.4f} | {r['AP50']*100:.4f} | {r['AP75']*100:.4f} | {r['actions']} | {r['confirmed_correct']} | {r['known_harm']} | {r['unconfirmed']} | {r['net50']} / {r['net75']} |")
    lines+=["",f"无损新增机会共 {opportunities} 图。覆盖、机会回收、匹配新增与损失和成本见 result.csv。"
            "关闭 JS 的臂复用权重和阈值，只重算融合、资格与收益。额外成本包含实测 Qwen 前向，"
            "关闭 JS 的 Qwen 成本与完整臂共享。","",
            f"流程计数为 {dict(flow)}。消息幅度、分歧与系数中位数为 {medians}。"]
    (OUT/"RESULT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(result,flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("stage",choices=("build","smoke","seal","train","fit-gain","lock","predict-all","seal-predictions","evaluate-all"))
    p.add_argument("--config",type=Path,default=CONFIG)
    p.add_argument("--shard",type=int,choices=range(SHARDS))
    p.add_argument("--images",type=Path)
    args=p.parse_args()
    torch.set_num_threads(3)
    OUT.mkdir(exist_ok=True)
    cfg=read(args.config)
    if args.stage=="build":
        if (OUT/"build_receipt.json").exists():
            raise RuntimeError("Build already sealed")
        prepare_resolved(cfg,args.config)
        return
    frozen_config(args.config)
    if args.stage=="smoke":
        from .smoke import smoke
        smoke(cfg)
    elif args.stage=="seal":
        seal()
    elif args.stage=="train":
        require_build()
        world=int(os.environ.get("WORLD_SIZE","1"))
        rank=int(os.environ.get("RANK","0"))
        if world>1:
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
            dist.init_process_group("nccl")
        try:
            train_semantic(read(INDEX),cfg,rank,world)
        finally:
            if world>1:
                dist.destroy_process_group()
    elif args.stage=="fit-gain":
        require_build()
        train_gain(read(INDEX),cfg)
    elif args.stage=="lock":
        lock_policy(cfg)
    elif args.stage=="predict-all":
        assert args.images is not None and args.shard is not None
        predict(cfg,args.images,args.shard)
    elif args.stage=="seal-predictions":
        seal_predictions()
    else:
        evaluate()


if __name__=="__main__":
    main()
