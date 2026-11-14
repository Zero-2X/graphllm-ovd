"""Enumerate fixed-box semantic edits, retaining same-class overlaps for supervision."""
import json

import numpy as np
import torch
from torchvision.ops import box_iou

from competition_cache import claim_targets
from counterfactual import gt_arrays, make_base, vocabulary
from dense_donor import ROOT, actual_donors, save_json
from operational_cache import sources

COLUMNS = ("image","query","original","proposed","donor","slot","score","aligned",
           "semantic_fix","semantic_bad","overlap_existing_class")


def eligible_actions(pack,base,nfine):
    if not len(pack["ids"]):
        return []
    owner = {}
    for slot,q in enumerate(base["query"].tolist()):
        owner.setdefault(q,slot)
    donor = actual_donors(pack,nfine)
    return [(q,c,d,owner[q]) for (q,c),d in donor.items()
            if q in owner and float(base["score"][owner[q]])<.35 and int(base["label"][owner[q]])!=c]


@torch.inference_mode()
def build(split):
    root,tag,_ = sources(split)
    manifest = json.loads((root/"manifest.json").read_text())
    names,nfine,_ = vocabulary(manifest,tag)
    assert nfine==48
    output = []
    for row in manifest["splits"][tag]:
        iid = row["id"]
        pack = torch.load(root/"cache"/tag/f"{iid}.pt",map_location="cpu",weights_only=True)
        if not len(pack["ids"]):
            continue
        base = make_base(pack,nfine)
        allowed = eligible_actions(pack,base,nfine)
        if not allowed:
            continue
        q,c,d,s = torch.tensor(allowed,dtype=torch.long).T
        old = base["label"][s]
        boxes = pack["boxes"][q].float()
        targets = claim_targets(row,manifest,names,boxes,torch.stack((old,c),dim=1))
        gt,leaf = gt_arrays(row,names,manifest)
        if len(gt):
            match,index = box_iou(boxes,gt).max(1)
            aligned = match>=.50
            fix = aligned & (leaf[index]==c)
            bad = aligned & (leaf[index]==old)
        else:
            aligned = fix = bad = torch.zeros(len(q),dtype=torch.bool)
        duplicate = ((box_iou(boxes,base["box"])>.50) & (c[:,None]==base["label"])).any(1)
        keep = (targets>=0).all(1)
        data = np.column_stack((np.full(len(q),iid),q,old,c,d,s,base["score"][s],aligned,fix,bad,duplicate))
        output.append(data[keep.numpy()])
    actions = np.concatenate(output)
    assert len(actions)==len(np.unique(actions[:,[0,1,3]],axis=0))
    assert not ((actions[:,8]>0)&(actions[:,9]>0)).any()
    path = ROOT/f"semantic_actions_{split}.npz"
    with path.with_suffix(".part").open("wb") as f:
        np.savez_compressed(f,rows=actions,columns=np.array(COLUMNS))
    path.with_suffix(".part").replace(path)
    dup = actions[:,10]>0
    fixes = actions[:,8]>0
    bad = actions[:,9]>0
    return dict(actions=len(actions),images=len(np.unique(actions[:,0])),
                fixes=int(fixes.sum()),old_harmed=int(bad.sum()),
                covered_actions=int(dup.sum()),covered_fixes=int((dup&fixes).sum()),
                covered_old_harmed=int((dup&bad).sum()),fix_images=len(np.unique(actions[fixes,0])))


if __name__=="__main__":
    torch.set_num_threads(2)
    result = {}
    for split in ("train","mine","cal"):
        result[split] = build(split)
        print(split,result[split],flush=True)
    save_json(ROOT/"semantic_action_supply.json",result)
