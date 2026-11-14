"""Materialize supervision from permitted frozen training/optimization caches."""
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch

from counterfactual import ImageReplay,candidate_actions,make_base,vocabulary,save_json

V1=Path(__file__).resolve().parent.parent/"hierarchical_instance_v1"
KINDS=("raise","replace","add_free","add_reallocate")
LEVELS=(.151,.351)
COLUMNS=("image","query","class","kind","native_slot","score_milli","receiver_truth",
         "receiver_proposal_correct","native_class_correct","independent_donor_correct",
         "claim_matched","gain","harm","semantic_change","new_at_035","lost_at_floor","fp_delta_at_035")


def export(split,manifest):
    names,nfine,_=vocabulary(manifest,split)
    result=[]
    summary=Counter()
    rows=manifest["splits"][split]
    for i,row in enumerate(rows,1):
        pack=torch.load(V1/"cache"/split/(str(row["id"])+".pt"),map_location="cpu",weights_only=True)
        base=make_base(pack,nfine)
        replay=ImageReplay(row,pack,base,manifest,names)
        if len(pack["ids"]):
            rec,edge=torch.nonzero(pack["valid"]&(pack["labels"]<nfine)&~pack["high"][:,None],as_tuple=True)
            edges={}
            for j,e in zip(rec.tolist(),edge.tolist()):
                key=(int(pack["ids"][j]),int(pack["labels"][j,e]))
                if key not in edges:
                    edges[key]=int(pack["node_ids"][pack["j"][j,e]])
            argmax=pack["logits"][:,:nfine].argmax(-1).tolist()
        for score in LEVELS:
            for action in candidate_actions(pack,base,nfine,trial_score=score):
                kind,q,c,slot=action
                new,lost,fp,sem,target=replay.delta(action,trial_score=score)
                near=int(replay.iou[q].argmax()) if len(replay.y) else -1
                gt=near if near>=0 and replay.iou[q,near]>=.50 else -1
                if gt<0 and len(replay.cy) and bool(((replay.cy==c)&(replay.ioa[q]>=.50)).any()):
                    summary["ignored_crowd"]+=1
                    continue
                receiver_correct=gt>=0 and replay.y[gt]==c
                own=np.flatnonzero(replay.bq==q)
                old=int(replay.bc[own[0]]) if len(own) else argmax[q]
                donor=edges[q,c]
                closest=int(replay.iou[donor].argmax()) if len(replay.y) else -1
                real=gt>=0 and closest>=0 and closest!=gt and replay.iou[donor,closest]>=.50 and replay.y[closest]==c
                gain=target>=0 and (any(new) or min(fp)<0 or sem>0)
                harm=any(lost) or max(fp)>0 or sem<0
                summary["rows"]+=1
                summary["gain"]+=int(gain)
                summary["harm"]+=int(harm)
                summary["safe_gain"]+=int(gain and not harm)
                summary["safe_class_fix"]+=int(gain and not harm and sem>0)
                summary["receiver_proposal_correct"]+=int(receiver_correct)
                summary["true_pair"]+=int(receiver_correct and real)
                summary["both_gain_harm"]+=int(gain and harm)
                summary[kind+"_rows"]+=1
                summary[kind+"_safe_gain"]+=int(gain and not harm)
                result.append((row["id"],q,c,KINDS.index(kind),-1 if slot is None else slot,
                               round(1000*score),gt,int(receiver_correct),int(gt>=0 and replay.y[gt]==old),
                               int(real),int(target>=0),int(gain),int(harm),sem,len(new[2]),len(lost[0]),fp[2]))
        if i%160==0 or i==len(rows):
            print(split,i,len(rows),flush=True)
    array=np.asarray(result,dtype=np.int32)
    assert len(array)==summary["rows"] and array.shape[1]==len(COLUMNS)
    path=Path("paired_targets_"+split+".npz")
    temp=path.with_suffix(".part")
    with temp.open("wb") as f:
        np.savez_compressed(f,rows=array,columns=np.array(COLUMNS))
    temp.replace(path)
    return dict(summary,images=len(rows),path=str(path))


if __name__=="__main__":
    manifest=json.loads((V1/"manifest.json").read_text())
    result={}
    for split in ("train","cal"):
        result[split]=export(split,manifest)
        save_json("paired_targets.json",result)
        print(split,result[split],flush=True)
