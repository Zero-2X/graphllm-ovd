"""Mine strictly verified, same-donor receiver contrasts from the frozen graph."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from layer_entropy_cache import encode, make_base
from objects365_cache import WideGraphPack, block_path, split_rows
from objects365_data import BLOCK, FOLDER, HERE, INDEX, SHARDS, digest, read
from objects365_donor_view_entropy import paired_donors
from objects365_fit import ImageReplay
from objects365_rare import RARE_INDEX

OUT = FOLDER/"receiver_pair_views"
PLAN = HERE/"OBJECTS365_RECEIVER_PAIR_VIEWS.md"


def blocks(doc, splits, shard):
    for split in splits:
        rare = split=="rare_train"
        src = read(RARE_INDEX) if rare else doc
        manifest = digest(RARE_INDEX if rare else INDEX)
        frames = split_rows(src,"train" if rare else split,shard)
        for start in range(0,len(frames),BLOCK):
            group = frames[start:start+BLOCK]
            part = start//BLOCK
            prior = block_path(split,shard,part)
            data = OUT/f"{split}_{shard}_{part:03d}.pt"
            images = [r["id"] for r in group]
            yield split,group,prior,data,dict(index=manifest,images=images)


def harmful(target):
    return target["wrong"] or any(target["lost"])


def candidates(record):
    classes = [int(c) for c in record["actions"][:,1]]
    positive = {c for c,t in zip(classes,record["outcomes"]) if t["correct"]}
    negative = {c for c,t in zip(classes,record["outcomes"]) if harmful(t)}
    shared = positive & negative
    return [i for i,(c,t) in enumerate(zip(classes,record["outcomes"]))
            if c in shared and (t["correct"] or harmful(t))]


def edge_rows(pack, actions):
    lookup = {int(query): i for i,query in enumerate(pack["ids"])}
    rows = torch.tensor([lookup[int(query)] for query in actions[:,0]])
    mask = pack["valid"][rows] & (pack["labels"][rows]==actions[:,1:2]) & ~pack["high"][rows,None]
    assert ((mask.sum(1)>=1)&(mask.sum(1)<=2)).all()
    return rows,mask


@torch.inference_mode()
def edge_features(reader, pack, base, actions, pooled, rows, mask, requests):
    result = pooled[requests[:,0]].clone()
    for edge in requests[:,1].unique().tolist():
        use = (requests[:,1]==edge)&(mask[requests[:,0]].sum(1)>1)
        indices = requests[use,0]
        if not len(indices):
            continue
        selected = rows[indices]
        valid = pack["valid"].clone()
        alternatives = mask[indices].clone()
        alternatives[:,edge] = False
        valid[selected] &= ~alternatives
        result[use] = encode(reader,dict(pack,valid=valid),base,actions[indices])[:,0]
    return result


def find_pairs(replay, pack, base, actions, rows, mask, outcomes, counts):
    groups = defaultdict(lambda:([],[]))
    for index,edge in torch.nonzero(mask).tolist():
        q,c,slot = actions[index].tolist()
        donor = int(pack["node_ids"][pack["j"][rows[index],edge]])
        if not replay.iou.shape[1]:
            continue
        qmatch,dmatch = replay.iou[[q,donor]].argmax(1)
        aligned = replay.iou[q,qmatch]
        if aligned<.50 or replay.iou[donor,dmatch]<.50 or qmatch==dmatch or replay.y[dmatch]!=c:
            continue
        correct = bool(outcomes[index]["correct"])
        if correct != (replay.y[qmatch]==c):
            continue
        area = float((pack["boxes"][q,2:]-pack["boxes"][q,:2]).prod())
        quality = np.array([aligned,float(base["score"][slot]),.1*np.log(max(area,1e-8))])
        groups[donor,c][int(correct)].append((index,edge,quality))
    pairs = []
    for neg,pos in groups.values():
        if not neg or not pos:
            continue
        distance = np.abs(np.stack([x[2] for x in pos])[:,None]-np.stack([x[2] for x in neg])[None]).sum(-1)
        used_pos,used_neg = set(),set()
        for flat in distance.argsort(axis=None,kind="stable"):
            i,j = np.unravel_index(flat,distance.shape)
            if i in used_pos or j in used_neg:
                continue
            used_pos.add(i)
            used_neg.add(j)
            pairs.append((pos[i][:2],neg[j][:2]))
    counts.update(verified_groups=sum(bool(a and b) for a,b in groups.values()),pairs=len(pairs))
    return pairs


@torch.inference_mode()
def extract(doc, frames, record_by_image, engine, mode, counts):
    records = {}
    for frame in frames:
        iid = frame["id"]
        ref = record_by_image.get(iid)
        if ref is None or "actions" not in ref:
            continue
        mining = mode in ("train","rare_train","monitor")
        indices = candidates(ref) if mining else list(range(len(ref["actions"])))
        if not indices:
            continue
        counts["attempted"] += 1
        with Image.open(frame["path"]) as image:
            pack,_,_ = engine.native_pack(image.convert("RGB"))
        base = make_base(pack,365)
        actions = ref["actions"][indices]
        q,_,slots = actions.T
        if int(slots.max())>=len(base["query"]) or not torch.equal(base["query"][slots],q):
            counts["mismatch"] += 1
            assert mining, (iid,"native slot")
            continue
        if "base" in ref:
            assert all(torch.equal(base[k],ref["base"][k]) for k in base) and torch.equal(pack["boxes"],ref["boxes"])
        if mining:
            rows,mask = edge_rows(pack,actions)
            replay = ImageReplay(frame,pack,base,doc,doc["names"])
            pairs = find_pairs(replay,pack,base,actions,rows,mask,[ref["outcomes"][i] for i in indices],counts)
            if pairs:
                requests = torch.tensor(pairs).reshape(-1,2)
                x = edge_features(engine.reader,pack,base,actions,ref["x"][indices,0],rows,mask,requests)
                donor = pack["node_ids"][pack["j"][rows[requests[:,0]],requests[:,1]]].reshape(-1,2)
                paired_actions = actions[requests[:,0]].reshape(-1,2,3)
                assert torch.equal(donor[:,0],donor[:,1]) and torch.equal(paired_actions[:,0,1],paired_actions[:,1,1])
                assert (paired_actions[:,0,0]!=paired_actions[:,1,0]).all()
                records[iid] = dict(x=x.reshape(-1,2,3584), action=paired_actions, donor=donor[:,0])
                counts["pair_images"] += 1
        else:
            selected,rows,edges,_,_ = paired_donors(pack,actions)
            if len(selected):
                views = []
                for m in range(2):
                    valid = pack["valid"].clone()
                    valid[rows,edges[:,1-m]] = False
                    views.append(encode(engine.reader,dict(pack,valid=valid),base,actions[selected])[:,0])
                records[iid] = dict(indices=selected, x=torch.stack(views,dim=1), actions=actions[selected])
                counts.update(multi=len(selected),multi_images=1)
    return records


def run(doc, shard, stage):
    engine = WideGraphPack(doc["names"],365,doc["parents"])
    splits = ("train","rare_train","monitor") if stage=="mine" else (stage,)
    if stage!="mine":
        assert read(OUT/"qualification.json")["passed"]
        if stage=="eval":
            assert (OUT/"policy.json").is_file()
    for split,frames,source,dest,source_lock in blocks(doc,splits,shard):
        anchor = dict(source=digest(source), data=source_lock, plan=digest(PLAN), code=digest(Path(__file__)))
        if dest.exists():
            assert torch.load(dest,map_location="cpu",weights_only=True)["lock"]==anchor
            continue
        cached = torch.load(source,map_location="cpu",weights_only=True)
        assert cached["lock"]==source_lock
        counts = Counter(images=len(frames))
        records = extract(doc,frames,{r["image"]:r for r in cached["records"]},engine,split,counts)
        temp = dest.with_suffix(".part")
        torch.save(dict(lock=anchor,records=records,counts=dict(counts)),temp)
        temp.replace(dest)
        print(dest.name,dict(counts),flush=True)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage",choices=("mine","policy","eval"))
    parser.add_argument("--shard",type=int,choices=range(SHARDS),required=True)
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    run(read(INDEX),args.shard,args.stage)


