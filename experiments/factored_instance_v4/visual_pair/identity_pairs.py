"""Index same- and distinct-instance overlaps on the actual weak-label action pool."""
import json

import numpy as np
import torch
from torchvision.ops import box_iou

from dense_donor import ROOT,save_json,vocabulary
from counterfactual import gt_arrays,make_base
from operational_cache import sources
from semantic_action_pool import COLUMNS
from train_factor import by_image

PAIR_FIELDS = ("image","query","class","donor","reference_query","slot","reference_slot","state",
               "receiver_gt","reference_gt","receiver_iou","reference_iou","pair_iou")


def identity_targets(row,manifest,names,boxes,query,ref,c):
    gt,y = gt_arrays(row,names,manifest)
    if not len(y):
        q = np.zeros(len(query),dtype=np.float32)
        return np.zeros(len(query),dtype=int),np.full(len(query),-1),np.full(len(query),-1),q,q.copy()
    overlap = box_iou(boxes,gt)
    mask = y[None]==torch.from_numpy(c)[:,None]
    p,i = overlap[query].masked_fill(~mask,-1).max(1)
    v,j = overlap[ref].masked_fill(~mask,-1).max(1)
    p,v = p.numpy(),v.numpy()
    i,j = i.numpy(),j.numpy()
    state = np.where((p>=.50)&(v>=.50),np.where(i==j,1,2),0)
    return state,np.where(p>=.50,i,-1),np.where(v>=.50,j,-1),p.clip(0),v.clip(0)


@torch.inference_mode()
def build(split):
    root,tag,_ = sources(split)
    manifest = json.loads((root/"manifest.json").read_text())
    names,nfine,_ = vocabulary(manifest,tag)
    assert nfine==48
    with np.load(ROOT/f"semantic_actions_{split}.npz") as raw:
        assert tuple(raw["columns"])==COLUMNS
        a = raw["rows"]
    index = by_image(a[a[:,10]>0])
    pieces = []
    identical = 0
    for row in manifest["splits"][tag]:
        iid = row["id"]
        if iid not in index:
            continue
        actions = index[iid]
        pack = torch.load(root/"cache"/tag/f"{iid}.pt",map_location="cpu",weights_only=True)
        base = make_base(pack,nfine)
        q,_,c,d,s = actions[:,1:6].astype(int).T
        assert np.array_equal(base["query"][s].numpy(),q)
        iou = box_iou(base["box"][s],base["box"])
        same_class = torch.from_numpy(c)[:,None]==base["label"]
        same_query = torch.from_numpy(q)[:,None]==base["query"]
        overlap = (iou>.50)&same_class
        identical += int((overlap&same_query).any(1).sum())
        a,b = torch.nonzero(overlap&~same_query,as_tuple=True)
        if not len(a):
            continue
        ref = base["query"][b].numpy()
        q,c,d,s = [x[a.numpy()] for x in (q,c,d,s)]
        state,g1,g2,u,v = identity_targets(row,manifest,names,pack["boxes"],q,ref,c)
        pieces.append(np.column_stack((np.full(len(q),iid),q,c,d,ref,s,b,state,g1,g2,u,v,iou[a,b])))
    data = np.concatenate(pieces)
    assert len(data)==len(np.unique(data[:,:5],axis=0))
    path = ROOT/f"identity_pairs_{split}.npz"
    with path.with_suffix(".part").open("wb") as out:
        np.savez_compressed(out,rows=data,columns=np.array(PAIR_FIELDS))
    path.with_suffix(".part").replace(path)
    states = data[:,7].astype(int)
    return dict(images=len(np.unique(data[:,0])),pairs=len(data),same_query_actions=identical,
                states=np.bincount(states,minlength=3).tolist(),
                different_above_070=int(((states==2)&(data[:,12]>=.70)).sum()),
                different_images=len(np.unique(data[states==2,0])))


if __name__=="__main__":
    torch.set_num_threads(2)
    result = {}
    for s in ("train","mine","cal"):
        result[s] = build(s)
        print(s,result[s],flush=True)
    save_json(ROOT/"identity_pair_supply.json",result)
