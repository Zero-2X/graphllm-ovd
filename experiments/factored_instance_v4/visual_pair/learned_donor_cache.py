"""Cache full-instance supervision from actual graph donors and native proposals."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torchvision.ops import box_iou, roi_align

from dense_donor import ProposalEngine, ROOT, SHARDS, V1, vocabulary, xyxy
from counterfactual import gt_arrays

SIZE = 5
SPLITS = {"train": V1, "mine_train": ROOT.parent/"mining", "cal": V1}


def region_features(field, scale, boxes):
    patch = roi_align(field, [boxes*scale], SIZE, sampling_ratio=2, aligned=True)
    return patch.flatten(2).transpose(1,2).contiguous()


def class_features(engine, raw):
    text = raw.encoder_last_hidden_state_text[0]
    return torch.stack([text[engine.native.tok[engine.native.cat==c]].mean(0)
                        for c in range(engine.nfine)])


def candidate_pool(raw, boxes):
    coords = raw.enc_outputs_coord_logits[0]
    anchor = xyxy(coords[torch.isfinite(coords).all(1)].sigmoid())
    anchor = anchor[((anchor[:,2:]-anchor[:,:2])>0).all(1)]
    return torch.cat((anchor,boxes)),len(anchor)


def fit_regions(boxes, nanchor, truth):
    overlap = box_iou(boxes,truth)
    best, target = overlap.max(1)
    whole = (best>=.50)[:,None] & (target[:,None]==torch.arange(len(truth),device=truth.device))
    area = (boxes[:,2:]-boxes[:,:2]).clamp_min(0).prod(1)
    gt_area = (truth[:,2:]-truth[:,:2]).clamp_min(0).prod(1)
    inter = (torch.minimum(boxes[:,None,2:],truth[None,:,2:])
             -torch.maximum(boxes[:,None,:2],truth[None,:,:2])).clamp_min(0).prod(2)
    part = (inter>=.80*area[:,None]) & (area[:,None]<=.55*gt_area)
    oversized = (inter>=.80*gt_area) & (gt_area<=.55*area[:,None])
    nonwhole = (best<.50)[:,None]
    valid_area = (area>0)[:,None]
    def choose(values, mask):
        score, idx = values.masked_fill(~mask,-torch.inf).max(0)
        return idx.masked_fill(~score.isfinite(),-1)
    origins = torch.arange(len(boxes),device=boxes.device)<nanchor
    found = torch.stack((
        choose(overlap,whole & origins[:,None]),
        choose(overlap,whole & ~origins[:,None]),
        choose(-(overlap-.60).abs(),whole),
        choose(overlap,part & nonwhole & valid_area),
        choose(overlap,oversized & nonwhole & valid_area),
    ),dim=1)
    return found.cpu().tolist()


def select_pairs(pack, indexed, pool, nanchor, truth, labels, parents, nfine):
    regions = fit_regions(pool,nanchor,truth)
    dids = torch.tensor(sorted({int(t[0]) for t in indexed}),device=pool.device)+nanchor
    close = box_iou(pool[dids],pool).cpu()
    donor_pos = {int(d-nanchor):j for j,d in enumerate(dids.tolist())}
    cy = labels.tolist()
    distance = torch.cdist((truth[:,:2]+truth[:,2:])/2,(truth[:,:2]+truth[:,2:])/2).cpu()
    choices = set()
    def add(d,c,t,region,s):
        if region>=0 and close[donor_pos[d],region]<.30:
            choices.add((d+nanchor,region,c,t,s))
    for d,c,s,t in indexed:
        d,c,s,t = map(int,(d,c,s,t))
        assert cy[s]==cy[t]==c and s!=t
        for region in regions[t][:3]:
            add(d,c,t,region,2)
        for region in regions[t][3:]:
            add(d,c,t,region,0)
    for d,c,s in sorted({tuple(map(int,a[:3])) for a in indexed}):
        other = [t for t,leaf in enumerate(cy) if leaf!=c and max(regions[t][:3])>=0]
        other.sort(key=lambda t: (parents[cy[t]]!=parents[c] or parents[c]<0,float(distance[s,t])))
        for t in other[:2]:
            for region in regions[t][:2]:
                add(d,c,t,region,1)

    # Real source mistakes with known geometry and a mismatched category.
    r,e = torch.nonzero(pack["valid"] & (pack["labels"]<nfine),as_tuple=True)
    q = pack["node_ids"][pack["j"][r,e]]
    qp = torch.unique(torch.stack((q,pack["labels"][r,e]),dim=1),dim=0)
    source_overlap = box_iou(pool[qp[:,0].to(pool.device)+nanchor],truth)
    iou, own = source_overlap.max(1)
    mismatched = (iou>=.50) & (qp[:,1].to(pool.device)!=labels[own])
    for j in torch.nonzero(mismatched).flatten().tolist():
        d,c = map(int,qp[j])
        s = int(own[j])
        targets = [t for t,leaf in enumerate(cy) if t!=s and leaf!=c and max(regions[t][:3])>=0]
        targets.sort(key=lambda t:(cy[t]!=cy[s],float(distance[s,t])))
        for t in targets[:1]:
            for region in regions[t][:2]:
                if region>=0 and box_iou(pool[d+nanchor:d+nanchor+1],pool[region:region+1]).item()<.30:
                    choices.add((d+nanchor,region,c,t,1))
    return sorted(choices)


@torch.inference_mode()
def cache(split, shard):
    root = SPLITS[split]
    manifest = json.loads((root/"manifest.json").read_text())
    names, nfine, parents = vocabulary(manifest,split)
    assert nfine==48
    all_pairs = np.load(ROOT/f"donor_index_{split}.npz")["pairs"]
    index = defaultdict(list)
    for row in all_pairs:
        index[int(row[0])].append(row[1:])
    images = [r for r in manifest["splits"][split] if r["id"] in index][shard::SHARDS]
    engine = ProposalEngine(names,nfine)
    banks, texts, records = [], [], []
    offset = 0
    for n,row in enumerate(images,1):
        iid = row["id"]
        pack = torch.load(root/"cache"/split/f"{iid}.pt",map_location="cpu",weights_only=True)
        im = Image.open(root/"images"/row["dataset"]/row["file_name"]).convert("RGB")
        boxes, logits, field, scale = engine.native.forward(im)
        assert torch.equal(boxes.cpu(),pack["boxes"]) and torch.equal(logits.cpu(),pack["logits"])
        raw = engine.native.last_output
        pool,nanchor = candidate_pool(raw,boxes)
        truth,labels = gt_arrays(row,names,manifest)
        selected = select_pairs(pack,index[iid],pool,nanchor,truth.cuda(),labels.cuda(),parents,nfine)
        if selected:
            records_image = torch.tensor(selected,dtype=torch.long)
            keys,inv = torch.unique(records_image[:,:2],sorted=True,return_inverse=True)
            roi = region_features(field,scale,pool[keys])
            assert torch.isfinite(roi).all()
            banks.append(roi.half().cpu())
            texts.append(class_features(engine,raw)[records_image[:,2]].half().cpu())
            records.append(torch.cat((inv+offset,records_image[:,2:],
                                      torch.full((len(selected),1),iid,dtype=torch.long)),dim=1))
            offset += len(keys)
        if n%25==0 or n==len(images):
            print("learned donor cache",split,shard,n,len(images),offset,flush=True)
    table = torch.cat(records)
    output = ROOT/f"learned_donor_cache_{split}_{shard}.pt"
    torch.save(dict(roi=torch.cat(banks),text=torch.cat(texts),pairs=table,
                    columns=("donor","receiver","class","truth","state","image")),output.with_suffix(".part"))
    output.with_suffix(".part").replace(output)
    print(split,shard,len(table),torch.bincount(table[:,4],minlength=3).tolist(),flush=True)


if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--split",choices=SPLITS,required=True)
    p.add_argument("--shard",type=int,choices=range(SHARDS),required=True)
    a = p.parse_args()
    cache(a.split,a.shard)
