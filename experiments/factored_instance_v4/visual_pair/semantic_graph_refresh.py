"""Recompute only the GraphLLM evidence using the exact unfiltered online action batch."""
import argparse
import json

import numpy as np
import torch

from dense_donor import ROOT, SHARDS, vocabulary
from counterfactual import make_base
from operational_cache import sources
from paired_hypotheses import RelabelEvidence
from semantic_action_pool import eligible_actions
from semantic_evidence import FIELDS,graph_inputs
from train_factor import by_image
from train_hypothesis import SharedHypothesisHead

NAME = "semantic_route_online"
PREFIX = "semantic_online_evidence"


@torch.inference_mode()
def refresh(split,shard):
    root,tag,_ = sources(split)
    manifest = json.loads((root/"manifest.json").read_text())
    names,nfine,parents = vocabulary(manifest,tag)
    assert nfine==48
    with np.load(ROOT/f"semantic_evidence_{split}_{shard}.npz") as raw:
        cache = {k:raw[k] for k in raw.files}
    assert tuple(cache["fields"])==FIELDS
    actions = cache["rows"]
    ix = by_image(np.column_stack((actions[:,0],np.arange(len(actions)))))
    reader = RelabelEvidence()
    reader.vocabulary(names,nfine,parents)
    state = torch.load(ROOT.parent/"extent/hypothesis_best.pt",map_location="cpu",weights_only=True)
    head = SharedHypothesisHead(state["width"]).cuda().eval()
    head.load_state_dict(state["model"])
    old_col,new_col = FIELDS.index("old_graph"),FIELDS.index("delta_graph")
    x = cache["evidence"]
    old = x[:,[old_col,new_col]].copy()
    count = 0
    for row in manifest["splits"][tag]:
        iid = row["id"]
        if iid not in ix:
            continue
        pack = torch.load(root/"cache"/tag/f"{iid}.pt",map_location="cpu",weights_only=True)
        base = make_base(pack,nfine)
        q,c,d,slot = np.array(eligible_actions(pack,base,nfine),dtype=np.int64).T
        roles = np.column_stack((base["label"][slot].numpy(),c))
        graph = graph_inputs(reader,head,pack,q,roles)
        idx = ix[iid][:,1].astype(int)
        lookup = {(a,b):i for i,(a,b) in enumerate(zip(q,c))}
        match = np.array([lookup[tuple(k.astype(int))] for k in actions[idx][:,[1,3]]])
        assert np.array_equal(d[match],actions[idx,4]) and np.array_equal(slot[match],actions[idx,5])
        vals = graph[match]
        x[idx,old_col] = vals[:,0]
        x[idx,new_col] = vals[:,1]-vals[:,0]
        count += len(idx)
    assert count==len(actions)
    path = ROOT/f"{PREFIX}_{split}_{shard}.npz"
    with path.with_suffix(".part").open("wb") as f:
        np.savez_compressed(f,**cache)
    path.with_suffix(".part").replace(path)
    delta = np.abs(x[:,[old_col,new_col]]-old)
    print("online graph evidence",split,shard,count,delta.max(0).tolist(),flush=True)


if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--split",choices=("train","mine","cal"),required=True)
    p.add_argument("--shard",type=int,choices=range(SHARDS),required=True)
    a = p.parse_args()
    refresh(a.split,a.shard)
