"""Single-action, calibration-only detector replay. No fitted gate or test images."""
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torchvision.ops import box_iou

HERE=Path(__file__).resolve().parent
V2=HERE.parent/"selective_instance_v2"
sys.path.insert(0,str(V2))
from selective import CAP,make_base,gt_arrays,vocabulary
from evaluate import coco_eval,metrics
from train_selective import save_json

IOUS=(.50,.75)
SCORES=(.05,.15,.35)
PROTECT=.35
TRIAL_SCORE=.351


def candidate_actions(pack,base,nfine,trial_score=TRIAL_SCORE):
    if not len(pack["ids"]):
        return
    r,e=torch.nonzero(pack["valid"]&(pack["labels"]<nfine)&~pack["high"][:,None],as_tuple=True)
    if not len(r):
        return
    bq=base["query"].numpy()
    bc=base["label"].numpy()
    scores=base["score"].numpy()
    q=pack["ids"][r].numpy()
    c=pack["labels"][r,e].numpy()
    distinct=set()
    tail=int(np.argmin(scores)) if len(scores)==CAP and scores.min()<PROTECT else None
    overlaps=box_iou(pack["boxes"][q].float(),base["box"]).numpy()
    for j,(query,label) in enumerate(zip(q,c)):
        query,label=int(query),int(label)
        key=(query,label)
        if key in distinct:
            continue
        distinct.add(key)
        own=np.flatnonzero(bq==query)
        same=own[bc[own]==label]
        if len(same):
            slot=int(same[0])
            if scores[slot]<min(PROTECT,trial_score):
                yield "raise",query,label,slot
            continue
        if len(own) and scores[own[0]]>=PROTECT:
            continue
        if ((overlaps[j]>.50)&(bc==label)).any():
            continue
        if len(own):
            yield "replace",query,label,int(own[0])
        elif len(scores)<CAP:
            yield "add_free",query,label,None
        elif tail is not None:
            yield "add_reallocate",query,label,tail


def crowds(row,manifest,names):
    ids={manifest["categories"][c]:i for i,c in enumerate(names) if c in manifest["categories"]}
    rows=[a for a in row["annotations"] if a["iscrowd"] and a["category_id"] in ids]
    boxes=torch.tensor([a["bbox"] for a in rows],dtype=torch.float32).reshape(-1,4)
    boxes[:,2:]+=boxes[:,:2]
    boxes/=torch.tensor([row["width"],row["height"],row["width"],row["height"]])
    return boxes,np.array([ids[a["category_id"]] for a in rows],dtype=int)


def overlaps_for_crowds(boxes,crowd):
    a=boxes[:,None]
    b=crowd[None]
    intersect=(torch.minimum(a[...,2:],b[...,2:])-torch.maximum(a[...,:2],b[...,:2])).clamp_min(0).prod(-1)
    return (intersect/(a[...,2:]-a[...,:2]).prod(-1).clamp_min(1e-8)).numpy()


class ImageReplay:
    def __init__(self,row,pack,base,manifest,names):
        self.base=base
        self.boxes=pack["boxes"].float()
        truth,y=gt_arrays(row,names,manifest)
        crowd,cy=crowds(row,manifest,names)
        self.y=y.numpy()
        self.cy=cy
        self.iou=box_iou(self.boxes,truth).numpy()
        self.ioa=overlaps_for_crowds(self.boxes,crowd)
        self.bq=base["query"].numpy()
        self.bc=base["label"].numpy()
        self.bs=base["score"].numpy()
        self.by_class={int(c):np.flatnonzero(self.bc==c) for c in np.unique(self.bc)}
        self.before={c:self.match_class(c,self.bq[slots],self.bs[slots])[0] for c,slots in self.by_class.items()}
        self.empty=[(set(),0) for _ in range(len(IOUS)*len(SCORES))]

    def match_class(self,c,query,score,track_last=False):
        g=np.flatnonzero(self.y==c)
        crowd=np.flatnonzero(self.cy==c)
        order=np.argsort(-score,kind="stable")
        query,score=query[order],score[order]
        overlap=self.iou[np.ix_(query,g)]
        ignore=self.ioa[np.ix_(query,crowd)]
        effect=[]
        target=-1
        tracked=int(np.flatnonzero(order==len(order)-1)[0]) if track_last else None
        for tau in IOUS:
            matched=np.full(len(query),-1,dtype=int)
            free=np.ones(len(g),dtype=bool)
            ignored=np.zeros(len(query),dtype=bool)
            for j in range(len(query)):
                if len(g) and free.any():
                    candidate=np.where(free,overlap[j],-1)
                    m=int(candidate.argmax())
                    if candidate[m]>=tau:
                        matched[j]=g[m]
                        free[m]=False
                        continue
                ignored[j]=bool(len(crowd) and ignore[j].max()>=tau)
            if tracked is not None and tau==IOUS[0]:
                target=int(matched[tracked])
            for cut in SCORES:
                active=score>=np.float32(cut)
                tp=set(matched[active&(matched>=0)].tolist())
                fp=int((active&(matched<0)&~ignored).sum())
                effect.append((tp,fp))
        return effect,target

    def delta(self,action,trial_score=TRIAL_SCORE):
        kind,query,c,slot=action
        cut_class=None if slot is None else int(self.bc[slot])
        classes=(c,) if cut_class is None or cut_class==c else (c,cut_class)
        new=[set() for _ in self.empty]
        lost=[set() for _ in self.empty]
        fp=[0 for _ in self.empty]
        target=-1
        for label in classes:
            b=self.by_class.get(label,np.empty(0,dtype=int))
            b=b[b!=slot] if slot is not None else b
            q,s=self.bq[b],self.bs[b]
            if label==c:
                q=np.append(q,query)
                s=np.append(s,np.float32(trial_score))
            before=self.before.get(label,self.empty)
            after,matched=self.match_class(label,q,s,track_last=(label==c))
            if label==c:
                target=matched
            for j,((bids,bfp),(aids,afp)) in enumerate(zip(before,after)):
                new[j]|=aids-bids
                lost[j]|=bids-aids
                fp[j]+=afp-bfp
        own=np.flatnonzero(self.bq==query)
        semantic=0
        if len(own) and self.iou.shape[1]:
            t=int(self.iou[query].argmax())
            if self.iou[query,t]>=.50:
                previous=int(self.bc[own[0]])
                remaining=own[own!=slot] if slot is not None else own
                chosen=c if not len(remaining) or self.bs[remaining[0]]<trial_score else int(self.bc[remaining[0]])
                semantic=int(chosen==self.y[t])-int(previous==self.y[t])
        return new,lost,fp,semantic,target

    def output(self,action=None,trial_score=TRIAL_SCORE):
        if action is None:
            return self.base
        _,query,c,slot=action
        mask=torch.ones(len(self.bq),dtype=torch.bool)
        if slot is not None:
            assert self.bs[slot]<PROTECT
            mask[slot]=False
        additions=dict(query=torch.tensor([query]),label=torch.tensor([c]),score=torch.tensor([trial_score]),
                       box=self.boxes[query:query+1])
        out={k:torch.cat((self.base[k][mask],additions[k])) for k in self.base}
        assert len(out["query"])<=CAP
        protected=torch.from_numpy(self.bs>=PROTECT)
        assert bool(mask[protected].all())
        for k in self.base:
            assert torch.equal(out[k][:-1],self.base[k][mask])
        return out


def predictions(row,out,catids):
    xywh=out["box"].clone()
    xywh[:,2:]-=xywh[:,:2]
    xywh*=torch.tensor([row["width"],row["height"],row["width"],row["height"]])
    return [dict(image_id=row["id"],category_id=catids[c],bbox=b,score=s)
            for c,b,s in zip(out["label"].tolist(),xywh.tolist(),out["score"].tolist())]


def audit(split,manifest):
    names,nfine,_=vocabulary(manifest,split)
    catids=[manifest["categories"][x] for x in names[:nfine]]
    rows=manifest["splits"][split]
    counts={k:Counter() for k in ("raise","replace","add_free","add_reallocate")}
    whole=Counter()
    raw,oracle=[],[]
    by_image=[]
    for idx,row in enumerate(rows,1):
        pack=torch.load(V2/"cache"/split/(str(row["id"])+".pt"),map_location="cpu",weights_only=True)
        base=make_base(pack,nfine)
        replay=ImageReplay(row,pack,base,manifest,names)
        best_key,best=None,None
        picked={}
        for action in candidate_actions(pack,base,nfine):
            kind=action[0]
            new,lost,fp,semantic,target=replay.delta(action)
            gain=any(new) or min(fp)<0
            harm=any(lost) or max(fp)>0 or semantic<0
            safe=(gain or semantic>0) and not harm and target>=0
            c=counts[kind]
            c["actions"]+=1
            c["gain"]+=int(gain and target>=0)
            c["gain_without_valid_proposal"]+=int(gain and target<0)
            c["harm"]+=int(harm)
            c["strict_benefit"]+=int(safe)
            c["classification_correction"]+=int(semantic>0)
            c["classification_regression"]+=int(semantic<0)
            c["strict_classification_correction"]+=int(safe and semantic>0)
            c["lost_true_at_floor"]+=int(bool(lost[0]))
            c["raises_false_positives_at_035"]+=int(fp[2]>0)
            if safe:
                picked.setdefault(kind,set()).update(new[2])
                # One truth-chosen action per image keeps this an interpretable opportunity test.
                key=(len(new[2]),len(new[5]),sum(map(len,new)),-sum(fp),semantic)
                if best_key is None or key>best_key:
                    best_key,best=key,action
        for kind,unique in picked.items():
            counts[kind]["images_with_strict_benefit"]+=1
            counts[kind]["unique_added_tp_at_035"]+=len(unique)
        if best is not None:
            whole["images_with_selected_oracle_action"]+=1
            whole["oracle_"+best[0]]+=1
            by_image.append(dict(image_id=row["id"],action=best[0],query=best[1],
                                 category=names[best[2]],removed_slot=best[3],priority=list(best_key)))
        raw.extend(predictions(row,base,catids))
        oracle.extend(predictions(row,replay.output(best),catids))
        if idx%80==0 or idx==len(rows):
            print(split,idx,len(rows),flush=True)
    original=coco_eval(rows,raw,catids)
    hypothetical=coco_eval(rows,oracle,catids)
    result=dict(images=len(rows),by_action={k:dict(v) for k,v in counts.items()},oracle=dict(whole),
                baseline=metrics(original),truth_selected_oracle=metrics(hypothetical),selected=by_image)
    save_json("verified_audit_"+split+".json",result)
    print(split,{k:v for k,v in result.items() if k!="selected"},flush=True)


if __name__=="__main__":
    manifest=json.loads((V2/"manifest.json").read_text())
    for split in ("cal_score","cal_policy"):
        audit(split,manifest)
