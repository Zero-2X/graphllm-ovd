"""Extract original and proposed category claims from the actual weak-box groups."""
import argparse
from collections import defaultdict
import json

import numpy as np
from PIL import Image
import torch
from torchvision.ops import box_iou

from dense_donor import ROOT, SHARDS, ProposalEngine, actual_donors, save_json, vocabulary
from counterfactual import crowds, gt_arrays, make_base, overlaps_for_crowds
from joint_gate import load,logit
from learned_donor_cache import class_features,region_features
from operational_cache import sources
from patch_consensus_train import PatchConsensusMatcher
from train_factor import by_image

FEATURES = ("native","graph","v5_alignment","patch_match","donor_present",
            "rival_native_delta","rival_alignment_delta","rival_match_delta","rivals")
KEYS = ("image","query","old","proposed","old_donor","proposed_donor")


def claim_targets(row,manifest,names,boxes,roles):
    truth,y = gt_arrays(row,names,manifest)
    iou = box_iou(boxes,truth)
    supported = ((iou[:,None,:]>=.50) & (roles[:,:,None]==y)).any(2)
    crowded,cy = crowds(row,manifest,names)
    ioa = torch.from_numpy(overlaps_for_crowds(boxes,crowded))
    ignore = ((ioa[:,None,:]>=.50) & (roles[:,:,None]==torch.from_numpy(cy))).any(2) & ~supported
    return supported.long().masked_fill(ignore,-1)


def group_features(features, q, classes, donors):
    # Compare with other receiver queries, including claims already named that class.
    flat = features.reshape(-1,features.shape[-1])
    ids = np.repeat(q,2)
    groups = defaultdict(list)
    for j,(c,d) in enumerate(zip(classes.ravel(),donors.ravel())):
        if d>=0:
            groups[(c,d)].append(j)
    extra = np.zeros((len(flat),4),dtype=np.float32)
    for group in groups.values():
        unique, first, inverse = np.unique(ids[group],return_index=True,return_inverse=True)
        if len(unique)<2:
            continue
        base = flat[np.asarray(group)[first]][:,[0,2,3]]
        order = np.argsort(-base,axis=0,kind="stable")
        cols = np.arange(3)
        rival = np.where(inverse[:,None]==order[0],base[order[1],cols],base[order[0],cols])
        extra[group,:3] = base[inverse]-rival
        extra[group,3] = np.log(len(unique))
    return np.concatenate((flat,extra),axis=1).reshape(len(q),2,len(FEATURES))


@torch.inference_mode()
def features(engine, model, pack, image, q, roles, donors, graph, spatial):
    boxes,logits,field,scale = engine.native.forward(image)
    assert torch.equal(boxes.cpu(),pack["boxes"]) and torch.equal(logits.cpu(),pack["logits"])
    value = absolute_features(engine,model,boxes,logits,field,scale,q,roles,donors,graph,spatial)
    return group_features(value,q,roles,donors)


@torch.inference_mode()
def absolute_features(engine,model,boxes,logits,field,scale,q,roles,donors,graph,spatial):
    references = np.where(donors>=0,donors,q[:,None])
    nodes,back = np.unique(np.concatenate((q,references.ravel())),return_inverse=True)
    roi = model.regions(region_features(field,scale,boxes[nodes]).half())
    query = roi[back[:len(q)]]
    source = roi[back[len(q):]]
    cue = model.cue(class_features(engine,engine.native.last_output).half())
    _, match = model.compare(source,query.repeat_interleave(2,0),cue[roles.ravel()])
    match = match.reshape(len(q),2).cpu().numpy()
    match[donors<0] = 0
    logits = logits.cpu().numpy()[q[:,None],roles]
    values = np.stack((logits,graph,np.repeat(spatial[:,None],2,1),match,donors>=0),axis=2)
    return values.astype(np.float32)


@torch.inference_mode()
def run(split,shard):
    root,tag,_ = sources(split)
    manifest = json.loads((root/"manifest.json").read_text())
    names,nfine,_ = vocabulary(manifest,tag)
    assert nfine==48
    data,cols = load(split,str(ROOT.parent/"extent"/"transport"))
    keys = by_image(data)
    rows = [r for r in manifest["splits"][tag] if r["id"] in keys][shard::SHARDS]
    engine = ProposalEngine(names,nfine)
    qualified = json.loads((ROOT/"patch_consensus_extended_qualification.json").read_text())
    assert qualified["passed"]
    saved = torch.load(ROOT/"patch_consensus_extended_final.pt",map_location="cpu",weights_only=True)
    assert saved["epoch"]==qualified["evaluation_epoch"]
    model = PatchConsensusMatcher().cuda().eval()
    model.load_state_dict(saved["model"])
    ids, targets, evidence = [], [], []
    for i,row in enumerate(rows,1):
        iid = row["id"]
        actions = keys[iid]
        pack = torch.load(root/"cache"/tag/f"{iid}.pt",map_location="cpu",weights_only=True)
        base = make_base(pack,nfine)
        owner = {}
        for query,c in zip(base["query"].tolist(),base["label"].tolist()):
            owner.setdefault(query,c)
        q,claim = actions[:,[cols["query"],cols["class"]]].astype(int).T
        old = np.array([owner[j] for j in q],dtype=np.int64)
        assert np.all(old!=claim)
        roles = np.column_stack((old,claim))
        graph_edges = actual_donors(pack,nfine)
        assert all((j,c) in graph_edges for j,c in zip(q,claim))
        donor = np.array([[graph_edges.get((j,c),-1) for c in labels] for j,labels in zip(q,roles)],dtype=np.int64)
        t = claim_targets(row,manifest,names,pack["boxes"][q],torch.from_numpy(roles)).numpy()
        assert np.all(t[actions[:,cols["strict_fix"]]>0,1]==1)
        assert np.all(t[actions[:,cols["old_harmed"]]>0,0]==1)
        graph = logit(torch.from_numpy(actions[:,[cols["old_prob"],cols["new_prob"]]])).numpy()
        spatial = logit(torch.from_numpy(actions[:,cols["v5_p50"]])).numpy()
        im = Image.open(root/"images"/row["dataset"]/row["file_name"]).convert("RGB")
        f = features(engine,model,pack,im,q,roles,donor,graph,spatial)
        assert np.isfinite(f).all()
        ids.append(np.column_stack((np.full(len(q),iid),q,roles,donor)))
        targets.append(t)
        evidence.append(f)
        if i%50==0 or i==1 or i==len(rows):
            print("competition cache",split,shard,i,len(rows),flush=True)
    index = np.concatenate(ids)
    t = np.concatenate(targets)
    x = np.concatenate(evidence)
    path = ROOT/f"competition_{split}_{shard}.npz"
    with path.with_suffix(".part").open("wb") as f:
        np.savez_compressed(f,keys=index,labels=t,evidence=x,fields=np.array(FEATURES),key_fields=np.array(KEYS))
    path.with_suffix(".part").replace(path)
    print(split,shard,len(index),"positives old/new",(t==1).sum(0).tolist(),flush=True)


if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--split",choices=("train","mine","cal"),required=True)
    p.add_argument("--shard",type=int,choices=range(SHARDS),required=True)
    a = p.parse_args()
    run(a.split,a.shard)
