"""Fixed one-slot policy and blind prediction. No label imports in this module."""
from time import perf_counter

import torch

from .common import OUT, HISTORY, SCHEMA, cache_write, config, data_lock, digest, load, locked_best, model_lock, read, save_json
from .model import Gain, gain_input
from .inputs import batch
from .native_bridge import NativeBridge
from graphllm_local_message_unique_v2 import run as old_run
from graphllm_local_message_unique_v2.core import pack_actions as old_pack


def key_index(values):
    return {v:i for i,v in enumerate(values)}


def one_slot(base,key,ci,check_duplicate=True):
    if key is None:
        return base
    _,q,slot,c0,c1=key
    assert q==int(base["query"][slot]) and ci[c0]==int(base["label"][slot]) and c0!=c1
    assert float(base["score"][slot])<.35
    if check_duplicate:
        assert not ((base["query"]==q)&(base["label"]==ci[c1])).any()
    label=base["label"].clone()
    label[slot]=ci[c1]
    assert int((label!=base["label"]).sum())==1
    return dict(base,label=label)


def select(records,policy):
    eligible=[r for r in records if r["semantic_eligible"]]
    if not eligible:
        return None,None
    chosen=min(eligible,key=lambda r:(-r["gain_logit"],tuple(r["key"])))
    chosen["image_top1"]=True
    cutoff=policy["gain_logit_cutoff"]
    if cutoff is None:
        chosen["reject_reason"]="policy_empty"
    elif chosen["gain_logit"]>=cutoff:
        chosen["executed"]=True
        chosen["reject_reason"]="executed"
        return chosen["key"],chosen
    else:
        chosen["reject_reason"]="below_gain_cutoff"
    return None,chosen


def score_actions(items,b,result,risk,gain,ids,cfg,off=False):
    z=result["base"] if off else result["final"]
    msg=torch.zeros_like(result["message"]) if off else result["message"]
    delta=torch.zeros_like(result["delta"]) if off else result["delta"]
    x=gain_input(b["native16"].cpu(),risk,z.cpu(),msg.cpu())
    logit=gain(x)
    finite=(torch.isfinite(x).all(1)&torch.isfinite(logit)&torch.isfinite(z.cpu()).all(1)&
            torch.isfinite(delta.cpu()).all(1)&torch.isfinite(result["base"].cpu()).all(1))
    if not off and "relation_probs" in result:
        for field in ("relation_probs","nonnull","gate","t"):
            finite&=torch.isfinite(result[field].cpu()).flatten(1).all(1)
    z=z.cpu()
    msg,delta=msg.cpu(),delta.cpu()
    base=result["base"].cpu()
    edge={k:result[k].cpu() for k in ("relation_probs","nonnull","gate","t","js","js_available") if k in result}
    has=b["edge_mask"].any(1).cpu()
    semantic=finite&has&(z[:,1]>z[:,0])&(z[:,1]>z[:,2])
    output=[]
    for i,(a,_,_) in enumerate(items):
        valid=bool(finite[i])
        n=len(a["donors"])
        rec=dict(key=a["key"],donor_query_ids=a["donor_query_ids"],donor_predicted_classes=a["donor_predicted_classes"],
                 has_raw_donor=bool(has[i]),all_finite=valid,is_executable=valid and bool(has[i]),
                 semantic_eligible=bool(semantic[i]),image_top1=False,executed=False,
                 mode="message_off" if off else "full",**ids)
        if valid:
            rec.update(z_base=base[i].tolist(),z_final=z[i].tolist(),delta=delta[i].tolist(),
                       applied_message128=msg[i].tolist(),risk_logit=float(risk[i]),gain_logit=float(logit[i]),
                       display_gain_sigmoid=float(logit[i].sigmoid()))
            if not off and "relation_probs" in edge:
                rec.update(relation_probs=edge["relation_probs"][i,:n].tolist(),
                           nonnull_mass=edge["nonnull"][i,:n].tolist(),gate=edge["gate"][i,:n].tolist(),
                           per_edge_message_norm=edge["t"][i,:n].norm(dim=1).tolist())
        rec["js_available"]=bool(not off and valid and n>=2 and edge["js_available"][i])
        rec["js_or_null"]=float(edge["js"][i]) if rec["js_available"] else None
        rec["reject_reason"]=("invalid_input" if not valid else "no_raw_donor" if not has[i]
                              else "not_image_top1" if semantic[i] else "semantic_not_c1")
        output.append(rec)
    return output


@torch.inference_mode()
def score_rows(model,gain,risk,items,ci,cfg,identity,off_too=False):
    result={k:[] for k in ("full","off") if k=="full" or off_too}
    reads=0
    for i in range(0,len(items),32):
        chunk=items[i:i+32]
        b=batch(chunk,ci,cfg,"cuda")
        sem=model(b,diagnose=True)
        rl=risk(b["native16"].cpu())
        reads+=len(chunk)+sum(1+len(a["donors"]) for a,_,_ in chunk if a["donors"])
        for k in result:
            result[k]+=score_actions(chunk,b,sem,rl,gain,identity,cfg,k=="off")
    return result,reads


def ready(cfg):
    from .bundle import construct

    artifacts=model_lock("policy")
    model=construct(cfg,"cuda","frozen").requires_grad_(False).eval()
    gain=Gain(cfg)
    gain.load_state_dict(locked_best("gain")["state"],strict=True)
    gain.requires_grad_(False).eval()
    return model,gain,artifacts


def freeze_ids(chain,policy_hash=None):
    return dict(feature_schema=SCHEMA,local_schema=config()["local"]["schema"],
                self_hash=chain["self"],message_hash=chain["message"],gain_hash=chain["gain"],policy_hash=policy_hash)


@torch.inference_mode()
def old_pick(model,head,policy,cfg,scan,cat):
    actions=scan["actions"]
    if not actions:
        return None
    packs=[old_pack(actions[i:i+32],"cuda") for i in range(0,len(actions),32)]
    zs,ds,rs,xs=[],[],[],[]
    for b in packs:
        with torch.autocast("cuda",dtype=torch.bfloat16,enabled=torch.cuda.is_bf16_supported()):
            z,d=model(b)
        zs.append(z.cpu())
        ds.append(d.cpu())
        rs.append(b["rho"].cpu())
        xs.append(b["native16"].cpu())
    z,d,r,x=[torch.cat(v) for v in (zs,ds,rs,xs)]
    winner,_=old_run.decide([a["key"] for a in actions],z,d,r,x,cfg["risk"](x).sigmoid(),
                           head,cfg["old"],cfg["old"]["message"]["lambda"],policy["gain_cutoff"])
    if winner is None or not winner["executed"]:
        return None
    k=winner["key"]
    return (*k[:3],cat[k[3]],cat[k[4]])


def old_bundle(doc,risk):
    from graphllm_local_message_unique_v2.train import fitted_gain,fitted_reader
    from graphllm_local_message_unique_v2.bridge import CONFIG as OLD_CFG

    prior=data_lock()["frozen_v1"]
    for name,sha in prior.items():
        if name not in ("predictions_manifest.json","result.json"):
            assert digest(HISTORY/name)==sha
    cfg=old_run.frozen_config(OLD_CFG)["config"]
    return fitted_reader(doc,cfg),fitted_gain(),read(HISTORY/"locked_policy.json"),dict(old=cfg,risk=risk)


@torch.inference_mode()
def predict(cfg,shard):
    doc=read(OUT/"images_holdout.json")
    assert digest(OUT/"images_holdout.json")==data_lock()["sets"]["holdout"]["blind"]
    assert all(set(f)=={"id","source_id","file_name","path","width","height","sha"} for f in doc["rows"])
    policy=read(OUT/"locked_policy.json")
    bridge=NativeBridge(doc,cfg)
    model,gain,chain=ready(cfg)
    assert policy["chain"]==chain and policy["mode"]=="full"
    old=old_bundle(doc,bridge.risk)
    cat=bridge.categories
    ci=key_index(cat)
    identity=freeze_ids(chain,digest(OUT/"locked_policy.json"))
    seal=dict(chain=chain,ids=identity,images=digest(OUT/"images_holdout.json"),
              v1=data_lock()["frozen_v1"])
    rows=doc["rows"][shard::cfg["workers"]]
    for start in range(0,len(rows),cfg["block"]):
        frames=rows[start:start+cfg["block"]]
        ids=[r["id"] for r in frames]
        path=OUT/f"predictions_{shard}_{start//cfg['block']:03}.pt"
        if path.exists():
            saved=load(path)
            assert saved["lock"]==seal and saved["images"]==ids
            continue
        output=[]
        for f in frames:
            time=perf_counter()
            torch.cuda.reset_peak_memory_stats()
            row,scan=bridge.scan(f)
            start_old=perf_counter()
            previous=old_pick(*old,scan,cat)
            torch.cuda.synchronize()
            old_time=perf_counter()-start_old
            start_new=perf_counter()
            items=[(a,row["nodes"],row["local"]) for a in row["actions"]]
            logs,reads=score_rows(model,gain,bridge.risk,items,ci,cfg,identity,off_too=True)
            full,_=select(logs["full"],policy)
            off,_=select(logs["off"],policy)
            keys=dict(native_D0=None,frozen_v1_js_on=previous,v2_full=full,v2_message_off=off)
            labels={k:one_slot(row["base"],v,ci)["label"] for k,v in keys.items()}
            torch.cuda.synchronize()
            timing=dict(row["timing"],v1=old_time,v2=perf_counter()-start_new,
                        total=perf_counter()-time,gpu_peak_bytes=torch.cuda.max_memory_reserved(),qwen_sequences=reads)
            output.append(dict(image=f["id"],base=row["base"],raw=row["raw"],rejected=row["rejected"],
                               labels=labels,keys=keys,records=logs,timing=timing))
        cache_write(path,dict(lock=seal,images=ids,rows=output))
        print("predict",shard,start,len(output),flush=True)


def seal_predictions(cfg):
    assert not (OUT/"predictions_manifest.json").exists()
    expected={r["id"] for r in read(OUT/"images_holdout.json")["rows"]}
    seen=set()
    hashes={}
    lock=None
    for p in sorted(OUT.glob("predictions_*_*.pt")):
        b=load(p)
        assert b["images"]==[r["image"] for r in b["rows"]]
        assert not seen.intersection(b["images"])
        seen.update(b["images"])
        if lock is None:
            lock=b["lock"]
        assert lock==b["lock"]
        hashes[p.name]=digest(p)
    assert seen==expected and len(seen)==cfg["evaluation"]["holdout_count"]
    assert lock["ids"]["policy_hash"]==digest(OUT/"locked_policy.json")
    save_json(OUT/"predictions_manifest.json",dict(lock=lock,hashes=hashes,images=len(seen)))
