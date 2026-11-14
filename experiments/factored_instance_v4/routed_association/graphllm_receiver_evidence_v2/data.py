"""Build spatial caches and separate supervision for the fixed train and monitor splits."""
from collections import Counter, defaultdict

import torch

from .common import OUT, annotated_split, cache_write, digest, feature_lock, load, new_inputs, read, vocabulary
from .native_bridge import NativeBridge, concrete_pool, raw_extension
from .targets import CONTRAST, INVALID, SUPPORT, annotate, weights


def alternative(action,target,pool):
    key=action["key"]
    original=target["relation_target"]
    relation=original[len(action["donors"]):]
    original=original[:len(action["donors"])]
    has_good=any(x in (SUPPORT,CONTRAST) for x in original)
    has_bad=INVALID in original
    for (j,q,edge),label in zip(pool,relation):
        useful=(label==INVALID and has_good or label in (SUPPORT,CONTRAST) and has_bad or
                label>=0 and not (has_good or has_bad))
        if useful:
            a=dict(action,donors=[j],edge=edge[None],donor_query_ids=[q],donor_predicted_classes=[key[4]])
            return a,label
    return None


def block_paths(split):
    return sorted(OUT.glob(f"visual_{split}_*.pt"))


@torch.inference_mode()
def cache(split,shard,cfg,limit=None):
    doc,frames=annotated_split(split)
    blind=new_inputs(doc,split)
    leaf=keys_index(vocabulary(doc)["text_to_canonical"])
    frames={f["id"]:f for f in frames}
    lock=feature_lock()
    population=blind["rows"][shard::cfg["workers"]]
    if limit is not None:
        population=population[:limit]
    bridge=NativeBridge(doc,cfg)
    for i in range(0,len(population),cfg["block"]):
        batch=population[i:i+cfg["block"]]
        tag=f"{split}_{shard}_{i//cfg['block']:03}"
        vp,tp=(OUT/f"{prefix}_{tag}.pt" for prefix in ("visual","targets"))
        ids=[f["id"] for f in batch]
        if vp.exists() and tp.exists():
            assert all(load(p)["lock"]==lock and load(p)["ids"]==ids for p in (vp,tp))
            continue
        inputs=[]
        targets={}
        n=0
        sums=torch.zeros(5,dtype=torch.float64)
        sq=sums.clone()
        counts=Counter()
        for safe in batch:
            f=frames[safe["id"]]
            row,scan=bridge.scan(safe,training_pool=split=="train")
            actions=row["actions"]
            pool=concrete_pool(row,scan["pack"],cfg) if split=="train" else None
            pools={a["key"]:raw_extension(row,a,scan["pack"],pool,cfg,leaf) for a in actions} if pool is not None else {}
            labels=annotate(f,doc,row,scan["pack"]["boxes"],
                            {a["key"]:a["donor_query_ids"]+[q for _,q,_ in pools.get(a["key"],())] for a in actions})
            useful=[]
            for a in actions:
                k=a["key"]
                t=labels[k]
                pair=alternative(a,t,pools[k]) if pools.get(k) else None
                t["relation_target"]=t["relation_target"][:len(a["donors"])]
                counts["original_actions"]+=1
                counts[f"semantic_{t['semantic_target']}"]+=1
                counts[f"gain_{t['gain_target']}"]+=1
                counts.update(f"relation_{y}" for y in t["relation_target"])
                if split=="train":
                    x=a["native"].double()
                    n+=1
                    sums+=x
                    sq+=x.square()
                if t["semantic_target"]>=0 or t["gain_target"]>=0 or any(y>=0 for y in t["relation_target"]):
                    targets[k,0]=t
                    useful.append(dict(a,view=0))
                if split=="train":
                    if pair is not None:
                        view,label=pair
                        if (k,0) not in targets:
                            targets[k,0]=t
                            useful.append(dict(a,view=0))
                        targets[k,1]=dict(t,relation_target=[label],training_pair_extension=True)
                        useful.append(dict(view,view=1))
                        counts["pair_extensions"]+=1
            counts.update(invalid=len(row["rejected"]))
            if useful:
                used=sorted({j for a in useful for j in [a["receiver"],*a["donors"]]})
                remap={j:z for z,j in enumerate(used)}
                for a in useful:
                    a["receiver"]=remap[a["receiver"]]
                    a["donors"]=[remap[j] for j in a["donors"]]
                inputs.append(dict(image=row["image"],nodes=row["nodes"][used],
                                   local=row["local"][used],actions=useful))
        cache_write(vp,dict(lock=lock,ids=ids,inputs=inputs,counts=dict(counts),norm=(n,sums,sq),
                            local_schema=bridge.local.identity))
        cache_write(tp,dict(lock=lock,ids=ids,targets=targets))
        print(split,shard,i,dict(counts),flush=True)


def keys_index(values):
    return {v:i for i,v in enumerate(values)}


def load_dataset(split,verify=True):
    lock=feature_lock()
    rows=[]
    labels={}
    hashes={}
    stats=Counter()
    n=0
    total=torch.zeros(5,dtype=torch.float64)
    square=total.clone()
    seen=set()
    local_schema=None
    for vp in block_paths(split):
        tp=OUT/vp.name.replace("visual_","targets_",1)
        v,t=load(vp),load(tp)
        assert v["lock"]==t["lock"]==lock and v["ids"]==t["ids"]
        assert not seen.intersection(v["ids"])
        seen.update(v["ids"])
        if local_schema is None:
            local_schema=v["local_schema"]
        assert local_schema==v["local_schema"]
        hashes.update({p.name:digest(p) for p in (vp,tp)})
        stats.update(v["counts"])
        n+=v["norm"][0]
        total+=v["norm"][1]
        square+=v["norm"][2]
        assert not (set(labels)&set(t["targets"]))
        labels.update(t["targets"])
        rows.extend(v["inputs"])
    blind=read(OUT/f"images_{split}.json")
    assert seen=={f["id"] for f in blind["rows"]},(split,len(seen))
    flat=[(a,im["nodes"],im["local"]) for im in rows for a in im["actions"]]
    assert len(flat)==len(labels)
    if verify and (OUT/"BUILD_COMPLETE.json").exists():
        expected=read(OUT/"BUILD_COMPLETE.json")["cache"][split]["hashes"]
        assert expected==hashes
    return flat,labels,dict(hashes=hashes,stats=dict(stats),norm=(n,total,square),local=local_schema,images=len(seen))


def supervision(rows,labels,stage,cfg):
    entries=[]
    states=defaultdict(list)
    for item in rows:
        a,_,_=item
        k=a["key"]
        t=labels[k,a["view"]]
        y=t["semantic_target"]
        rel=t["relation_target"]
        assert len(rel)==len(a["donors"])
        known=[j for j,r in enumerate(rel) if r>=0]
        sem=y>=0 and (stage=="self" or (len(known)==len(rel) and any(r in (SUPPORT,CONTRAST) for r in rel)))
        bad=INVALID in rel
        if stage=="self" and (a["view"] or not sem):
            continue
        if stage=="message" and not known:
            continue
        i=len(entries)
        entries.append(dict(item=item,y=y,relation=rel,sem_weight=0.,rel_weight=[0.]*len(rel),bad_weight=0.))
        if sem:
            states["sem"].append((y,k,i,None))
        if stage=="message":
            states["rel"].extend((rel[j],k,i,j) for j in known)
            if bad:
                states["bad"].append((0,k,i,None))
    for field in (("sem",) if stage=="self" else ("sem","rel","bad")):
        atoms=states[field]
        required=(0,) if field=="bad" else (0,1,2)
        w=weights([(s,k) for s,k,_,_ in atoms],required)
        for (_,_,i,j),v in zip(atoms,w):
            if j is None:
                entries[i][field+"_weight"]=v
            else:
                entries[i]["rel_weight"][j]=v
    groups=defaultdict(set)
    for r in entries:
        a=r["item"][0]
        for d,state in zip(a["donor_query_ids"],r["relation"]):
            if a["view"]==0 and state in (SUPPORT,CONTRAST):
                groups[a["key"][0],d,a["key"][4]].add(state)
    summary=dict(views=len(entries),actions=len({r["item"][0]["key"] for r in entries}),
                 images=len({r["item"][0]["key"][0] for r in entries}),
                 alternate=sum(r["item"][0]["view"] for r in entries),
                 fixed_donor_contrast=sum(len(v)==2 for v in groups.values()))
    for field,atoms in states.items():
        summary[field]={s:dict(atoms=sum(t[0]==s for t in atoms),
                               images=len({t[1][0] for t in atoms if t[0]==s}))
                        for s in sorted({t[0] for t in atoms})}
    return entries,summary


def batch_supervision(rows,slots,device):
    relation=torch.full((len(rows),slots),-1,dtype=torch.long)
    rel_weight=torch.zeros((len(rows),slots))
    for i,r in enumerate(rows):
        n=len(r["relation"])
        relation[i,:n]=torch.tensor(r["relation"],dtype=torch.long)
        rel_weight[i,:n]=torch.tensor(r["rel_weight"])
    output=dict(y=torch.tensor([r["y"] for r in rows]),relation=relation,
                sem_weight=torch.tensor([r["sem_weight"] for r in rows]),rel_weight=rel_weight,
                bad_weight=torch.tensor([r["bad_weight"] for r in rows]))
    return {k:v.to(device) for k,v in output.items()}
