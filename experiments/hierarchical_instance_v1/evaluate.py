"""Locked COCO evaluation and paired receiver diagnostics."""
import contextlib
import io
import json
import os
from collections import Counter
from pathlib import Path
import sys

import numpy as np
import torch
from torchvision.ops import batched_nms, box_iou

from cache import gt_arrays, measure_training, vocabulary
from model import StarAdapter, receiver_batch
from train import save_json

sys.path.insert(0,str(Path(os.environ["BASE"])/"experiments/omni_standard_v4/deps"))
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def detections(row,boxes,scores,catids):
    scores=scores.sigmoid()
    valid=((boxes[:,2:]-boxes[:,:2])>0).all(-1)
    i,c=torch.nonzero((scores>=.05)&valid[:,None],as_tuple=True)
    if not len(i):
        return []
    vals=scores[i,c]
    coords=boxes[i]
    keep=batched_nms(coords,vals,c,.50)[:100]
    xywh=coords[keep].clone()
    xywh[:,2:]-=xywh[:,:2]
    xywh*=torch.tensor([row["width"],row["height"],row["width"],row["height"]])
    return [dict(image_id=row["id"],category_id=catids[label],bbox=b,score=v)
            for b,v,label in zip(xywh.tolist(),vals[keep].tolist(),c[keep].tolist())]


def coco_eval(rows,results,catids,annotations=None):
    images=[{k:r[k] for k in ("id","file_name","width","height")} for r in rows]
    anns=[a for r in rows for a in r["annotations"]] if annotations is None else annotations
    ground=COCO()
    ground.dataset={"info":{},"images":images,"annotations":anns,
                    "categories":[dict(id=i,name=str(i)) for i in catids]}
    with contextlib.redirect_stdout(io.StringIO()):
        ground.createIndex()
        out=ground.loadRes(results) if results else COCO()
        if not results:
            out.dataset=dict(images=images,annotations=[],categories=ground.dataset["categories"])
            out.createIndex()
        ev=COCOeval(ground,out,"bbox")
        ev.params.imgIds=[r["id"] for r in rows]
        ev.params.catIds=catids
        ev.evaluate()
        ev.accumulate()
    return ev


def metrics(ev,base_ids=None):
    p=ev.eval["precision"][:,:,:,0,-1]
    def mean(v):
        x=v[v>=0]
        return float(x.mean()) if len(x) else None
    m=dict(AP=mean(p),AP50=mean(p[0]),AP75=mean(p[5]))
    if base_ids is not None:
        mask=np.array([i in base_ids for i in ev.params.catIds])
        m.update(AP_base=mean(p[:,:,mask]),AP_novel=mean(p[:,:,~mask]),
                 AP50_base=mean(p[0,:,mask]),AP50_novel=mean(p[0,:,~mask]))
    return m


def resampled_ap(ev,counts):
    """Reweight image-level matches, retaining the evaluator's stable score order."""
    n=len(ev.params.imgIds)
    nareas=len(ev.params.areaRng)
    recalls=ev.params.recThrs
    total=np.zeros(len(counts))
    used=np.zeros(len(counts))
    for k in range(len(ev.params.catIds)):
        entries=ev.evalImgs[k*nareas*n:k*nareas*n+n]
        positives=np.array([0 if e is None else np.count_nonzero(np.asarray(e["gtIgnore"])==0) for e in entries])
        n_gt=counts@positives
        active=np.flatnonzero(n_gt)
        used[active]+=1
        available=[(i,e) for i,e in enumerate(entries) if e is not None and len(e["dtScores"])]
        if not available or not len(active):
            continue
        order=np.argsort(-np.concatenate([e["dtScores"] for _,e in available]),kind="mergesort")
        ims=np.concatenate([np.full(len(e["dtScores"]),i,dtype=int) for i,e in available])[order]
        match=np.concatenate([e["dtMatches"]>0 for _,e in available],axis=-1)[:,order]
        good=np.concatenate([e["dtIgnore"] for _,e in available],axis=-1)[:,order]==0
        tp=match&good
        fp=~match&good
        for z in active:
            w=counts[z,ims][None,:]
            # Remove observations from unsampled images before precision interpolation.
            present=w[0]>0
            if not present.any():
                continue
            t=np.cumsum((tp*w)[:,present],axis=1)
            f=np.cumsum((fp*w)[:,present],axis=1)
            pr=np.maximum.accumulate((t/np.maximum(t+f,np.spacing(1)))[:,::-1],axis=1)[:,::-1]
            ix=np.stack([np.searchsorted(t[s]/n_gt[z],recalls,side="left") for s in range(len(t))])
            vals=np.take_along_axis(pr,ix.clip(max=pr.shape[-1]-1),axis=-1)
            total[z]+=(vals*(ix<pr.shape[-1])).mean()
    return total/used.clip(min=1)


def ap_interval(a,b):
    assert a.params.imgIds==b.params.imgIds
    n=len(a.params.imgIds)
    all_images=np.ones((1,n),dtype=np.int64)
    assert np.allclose(resampled_ap(a,all_images)[0],metrics(a)["AP"],atol=2e-6)
    assert np.allclose(resampled_ap(b,all_images)[0],metrics(b)["AP"],atol=2e-6)
    rng=np.random.default_rng(20261110)
    counts=rng.multinomial(n,np.full(n,1/n),size=200)
    diff=resampled_ap(b,counts)-resampled_ap(a,counts)
    return [float(x) for x in np.quantile(diff,[.025,.975])]


def fallback(logits,nfine,parents):
    scores=logits.sigmoid()
    top,leaf=scores[:,:nfine].max(-1)
    p=torch.tensor(parents)
    siblings=(p[None]==p[leaf,None])&(p[leaf,None]>=0)
    siblings[torch.arange(len(leaf)),leaf]=False
    runner=scores[:,:nfine].masked_fill(~siblings,0).amax(-1)
    ps,pi=scores[:,nfine:].max(-1)
    return torch.where((top>=.35)&((top-runner)>=.10),leaf,torch.where(ps>=.30,pi+nfine,-1))


def diagnostic(row,pack,new_logits,new_boxes,names,nfine,parents,manifest):
    ids,scope=measure_training(row,pack,names,nfine,parents,manifest)
    out=Counter(scope)
    if not len(ids):
        return out
    chosen=pack["ids"][ids]
    old_logits=pack["logits"][chosen].float()
    b0=pack["boxes"][chosen].float()
    b1=new_boxes[chosen]
    truth,y=gt_arrays(row,names,manifest)
    match=box_iou(b0,truth).argmax(-1)
    y=y[match]
    truth=truth[match]
    prev=old_logits[:,:nfine].argmax(-1)==y
    new=new_logits[chosen,:nfine].argmax(-1)==y
    i0=box_iou(b0,truth).diagonal()
    i1=box_iou(b1,truth).diagonal()
    out.update(matched=len(y),fine_before=int(prev.sum()),fine_after=int(new.sum()),
               fine_recovered=int((new&~prev).sum()),fine_spoiled=int((prev&~new).sum()),
               iou_before=float(i0.sum()),iou_after=float(i1.sum()),
               loc_changed=int(((b1-b0).abs().amax(-1)>1e-6).sum()),
               iou_gain_05=int(((i1-i0)>=.05).sum()),iou_drop_05=int(((i0-i1)>=.05).sum()))
    for arm,z in (("dino",old_logits),("graph",new_logits[chosen])):
        pick=fallback(z,nfine,parents)
        father=torch.tensor(parents)[y]
        out[arm+"_fine_emitted"]=int(((pick>=0)&(pick<nfine)).sum())
        out[arm+"_fine_correct"]=int((pick==y).sum())
        out[arm+"_parent_emitted"]=int((pick>=nfine).sum())
        out[arm+"_parent_correct"]=int(((pick>=nfine)&(pick==father)).sum())
        out[arm+"_unresolved"]=int((pick<0).sum())
    for name in ("leaf_opportunity","parent_opportunity","loc_opportunity"):
        out[name+"_images"]=int(scope[name]>0)
    return out


@torch.inference_mode()
def main():
    qualification=json.loads(Path("qualification.json").read_text())
    if not qualification["passed"]:
        raise SystemExit("Qualification failed. Efficacy remains unopened.")
    manifest=json.loads(Path("manifest.json").read_text())
    rows=manifest["splits"]["eval"]
    names,nfine,parents=vocabulary(manifest,"eval")
    catids=[manifest["categories"][c] for c in names[:nfine]]
    baseids={manifest["categories"][c] for c in manifest["base"]}
    parentids=list(range(1001,1001+len(names)-nfine))
    model=StarAdapter().cuda().eval()
    model.set_vocabulary(names,nfine,"cuda")
    state=torch.load("best.pt",weights_only=True,map_location="cpu")
    missing,extra=model.load_state_dict(state,strict=False)
    assert not extra and all(k.startswith("qwen.") for k in missing)
    fine,coarse={},{}

    for arm in ("dino","graph"):
        fine[arm]={}
        coarse[arm]={}
    audits={}
    for pos,row in enumerate(rows,1):
        pack=torch.load(Path("cache/eval")/(str(row["id"])+".pt"),map_location="cpu",weights_only=True)
        base=pack["logits"].float()
        orig=pack["boxes"].float()
        updated=base.clone()
        boxes=orig.clone()
        for r in torch.arange(len(pack["ids"])).split(64):
            if not len(r):
                continue
            batch={k:v.cuda() for k,v in receiver_batch(pack,r).items()}
            s,b=model(batch)
            updated[pack["ids"][r]]=s.cpu()
            boxes[pack["ids"][r]]=b.cpu()
        for arm,z,b in (("dino",base,orig),("graph",updated,boxes)):
            fine[arm][row["id"]]=detections(row,b,z[:,:nfine],catids)
            coarse[arm][row["id"]]=detections(row,b,z[:,nfine:],parentids)
        audits[row["id"]]=diagnostic(row,pack,updated,boxes,names,nfine,parents,manifest)
        if pos%100==0 or pos==len(rows):
            print("inference",pos,len(rows),flush=True)
    result={}
    for group,items in (("random",[r for r in rows if r["stratum"]=="random"]),
                        ("enriched",[r for r in rows if r["stratum"]!="random"])):
        item_ids={r["id"] for r in items}
        parentmap={catids[i]:parentids[p-nfine] for i,p in enumerate(parents) if p>=nfine}
        parent_anns=[dict(a,category_id=parentmap[a["category_id"]]) for r in items for a in r["annotations"]
                     if a["category_id"] in parentmap]
        evaluators={}
        group_result={}
        for arm in fine:
            detections_f=[v for i in item_ids for v in fine[arm][i]]
            detections_p=[v for i in item_ids for v in coarse[arm][i]]
            evaluators[arm]=coco_eval(items,detections_f,catids)
            group_result[arm]=metrics(evaluators[arm],baseids)
            group_result[arm]["parents"]=metrics(coco_eval(items,detections_p,parentids,parent_anns))
        group_result["ap_delta_ci95"]=ap_interval(evaluators["dino"],evaluators["graph"])
        diagnostics=Counter()
        for i in item_ids:
            diagnostics.update(audits[i])
        group_result["diagnostics"]=dict(diagnostics,images=len(item_ids))
        result[group]=group_result
        save_json("result.json",result)
        print(group,group_result,flush=True)


if __name__=="__main__":
    main()
