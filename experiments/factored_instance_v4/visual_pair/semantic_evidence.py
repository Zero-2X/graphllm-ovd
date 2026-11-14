"""Build paired semantic evidence and exact preservation targets for the expanded pool."""
import argparse
import json

import numpy as np
from PIL import Image
import torch
from torchvision.ops import box_iou

from competition_cache import FEATURES, actual_donors, claim_targets, features, group_features, logit
from competition_train import load_claims
from counterfactual import ImageReplay,make_base
from dense_donor import ProposalEngine, ROOT, SHARDS, save_json, vocabulary
from operational_cache import sources
from paired_hypotheses import RelabelEvidence,hypotheses
from patch_consensus_train import PatchConsensusMatcher
from relabel import read_factor
from semantic_action_pool import COLUMNS
from train_factor import by_image
from train_hypothesis import SharedHypothesisHead

FIELDS = (*("old_"+k for k in FEATURES),*("delta_"+k for k in FEATURES),
          "old_other_native","new_other_native","new_same_query_native")


class Evidence:
    def __init__(self,names,nfine,parents):
        self.engine = ProposalEngine(names,nfine)
        self.reader = RelabelEvidence()
        self.reader.vocabulary(names,nfine,parents)
        self.alignment = read_factor(ROOT.parent/"pooled/best.pt")
        state = torch.load(ROOT.parent/"extent/hypothesis_best.pt",map_location="cpu",weights_only=True)
        self.semantic = SharedHypothesisHead(state["width"]).cuda().eval()
        self.semantic.load_state_dict(state["model"])
        self.match = load_patch_model()

    @torch.inference_mode()
    def missing(self,pack,image,q,roles,donor):
        graph,spatial = paired_inputs(self.reader,self.alignment,self.semantic,pack,q,roles)
        return features(self.engine,self.match,pack,image,q,roles,donor,graph,spatial)[:,:,:5]


def load_patch_model():
    qual = json.loads((ROOT/"patch_consensus_extended_qualification.json").read_text())
    patch = torch.load(ROOT/"patch_consensus_extended_final.pt",map_location="cpu",weights_only=True)
    assert qual["passed"] and qual["evaluation_epoch"]==patch["epoch"]
    model = PatchConsensusMatcher().cuda().eval()
    model.load_state_dict(patch["model"])
    return model


@torch.inference_mode()
def paired_inputs(reader,alignment,semantic,pack,q,roles):
    index = {int(a):j for j,a in enumerate(pack["ids"])}
    grid,own = reader.spatial(pack,torch.tensor([index[int(j)] for j in q]))
    p50 = alignment.alignment_logits(grid.cuda(),own.cuda()).softmax(-1)[:,1:].sum(1).cpu()
    return graph_inputs(reader,semantic,pack,q,roles),logit(p50).numpy()


@torch.inference_mode()
def graph_inputs(reader,semantic,pack,q,roles):
    pair = hypotheses(reader,pack,torch.from_numpy(q),torch.from_numpy(roles))
    assert pair["supported"][:,1].all()
    pred = semantic({k:v.cuda() for k,v in pair.items()},torch.arange(len(q),device="cuda")).softmax(-1).cpu()
    return logit(pred[:,1:]).numpy()


def baseline_evidence(boxes,base,q,roles):
    iou = box_iou(boxes,base["box"])
    roles = torch.from_numpy(roles)
    same = roles[:,:,None]==base["label"]
    own = torch.from_numpy(q)[:,None,None]==base["query"]
    score = base["score"]
    other = (iou[:,None,:]*score).masked_fill(~(same&~own&(iou[:,None,:]>.50)),0).amax(2)
    self_new = (same[:,1]&own[:,0]).any(1)
    return other.numpy(),self_new.numpy()


def floor_losses(row,manifest,names,pack,base,actions,claims):
    # If neither class can match the box at IoU50, this edit cannot displace an IoU50 true positive.
    overlap = np.flatnonzero((claims==1).any(1))
    loss = np.zeros(len(actions),dtype=np.int16)
    if len(overlap):
        replay = ImageReplay(row,pack,base,manifest,names)
        for j in overlap:
            _,q,old,proposed,d,slot = map(int,actions[j,:6])
            _,lost,_,_,_ = replay.delta(("replace",q,proposed,slot),float(actions[j,6]))
            loss[j] = len(lost[0])
    return loss


@torch.inference_mode()
def run(split,shard):
    root,tag,_ = sources(split)
    manifest = json.loads((root/"manifest.json").read_text())
    names,nfine,parents = vocabulary(manifest,tag)
    assert nfine==48
    with np.load(ROOT/f"semantic_actions_{split}.npz") as data:
        assert tuple(data["columns"])==COLUMNS
        index = by_image(data["rows"])
    existing = load_claims((split,))
    lookup = {tuple(k):j for j,k in enumerate(existing.keys[:,[0,1,3]])}
    rows = [r for r in manifest["splits"][tag] if r["id"] in index][shard::SHARDS]
    engine = Evidence(names,nfine,parents)
    records,evidence,losses,independent_overlap,same_query = [],[],[],[],[]
    new_count,reused = 0,0
    for i,row in enumerate(rows,1):
        iid = row["id"]
        a = index[iid]
        assert (a[:,0]==iid).all()
        q,old,c,source,slot = a[:,1:6].astype(np.int64).T
        roles = np.column_stack((old,c))
        pack = torch.load(root/"cache"/tag/f"{iid}.pt",map_location="cpu",weights_only=True)
        base = make_base(pack,nfine)
        assert np.array_equal(base["query"][slot].numpy(),q) and np.array_equal(base["label"][slot].numpy(),old)
        assert np.array_equal(base["score"][slot].numpy(),a[:,6])
        graph = actual_donors(pack,nfine)
        donor = np.array([[graph.get((j,o),-1),graph[j,n]] for j,o,n in zip(q,old,c)])
        assert np.array_equal(donor[:,1],source)
        cached = np.array([lookup.get((iid,j,n),-1) for j,n in zip(q,c)])
        present = cached>=0
        absolute = np.empty((len(a),2,5),dtype=np.float32)
        if present.any():
            cached_keys = existing.keys[cached[present]]
            assert np.array_equal(cached_keys[:,[2,3]],roles[present])
            assert np.array_equal(cached_keys[:,4:6],donor[present])
            absolute[present] = existing.x[cached[present],:,:5]
            reused += int(present.sum())
        missing = ~present
        if missing.any():
            image = Image.open(root/"images"/row["dataset"]/row["file_name"]).convert("RGB")
            absolute[missing] = engine.missing(pack,image,q[missing],roles[missing],donor[missing])
            new_count += int(missing.sum())
        f = group_features(absolute,q,roles,donor)
        boxes = pack["boxes"][q]
        native,same = baseline_evidence(boxes,base,q,roles)
        covered = native[:,1]>0
        assert np.array_equal(covered|same,a[:,10]>0)
        claims = claim_targets(row,manifest,names,boxes,torch.from_numpy(roles))
        loss = floor_losses(row,manifest,names,pack,base,a,claims.numpy())
        x = np.column_stack((f[:,0],f[:,1]-f[:,0],native,same)).astype(np.float32)
        assert x.shape==(len(a),len(FIELDS)) and np.isfinite(x).all()
        records.append(a)
        evidence.append(x)
        losses.append(loss)
        independent_overlap.append(covered)
        same_query.append(same)
        if i%50==0 or i in (1,len(rows)):
            print("semantic features",split,shard,i,len(rows),"reused",reused,"new",new_count,flush=True)
    a,x,l = map(np.concatenate,(records,evidence,losses))
    other = np.concatenate(independent_overlap)
    same = np.concatenate(same_query)
    assert len(a)==reused+new_count
    path = ROOT/f"semantic_evidence_{split}_{shard}.npz"
    with path.with_suffix(".part").open("wb") as f:
        np.savez_compressed(f,rows=a,fields=np.array(FIELDS),evidence=x,lost_floor=l,
                            other_native=other,same_query=same,reused=reused,new=new_count)
    path.with_suffix(".part").replace(path)
    print(split,shard,len(a),int(l.sum()),flush=True)


if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--split",choices=("train","mine","cal"),required=True)
    p.add_argument("--shard",type=int,choices=range(SHARDS),required=True)
    args = p.parse_args()
    run(args.split,args.shard)
