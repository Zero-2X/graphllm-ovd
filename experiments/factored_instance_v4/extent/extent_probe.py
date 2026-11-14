"""Probe receiver completeness using matched masks from tight and wide prompts."""
from collections import Counter
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
from transformers import SamModel, SamProcessor

HERE=Path(__file__).resolve().parent
RELABEL=HERE.parent/"relabel"
sys.path.insert(0,str(RELABEL))
from relabel import V3
from counterfactual import save_json
from diagnose_factor import auc


def subgroup(box, boxes):
    inter=(torch.minimum(box[2:],boxes[:,2:])-torch.maximum(box[:2],boxes[:,:2])).clamp_min(0).prod(-1)
    area=(box[2:]-box[:2]).clamp_min(0).prod()
    ref=(boxes[:,2:]-boxes[:,:2]).clamp_min(0).prod(-1)
    if ((inter/(area+ref-inter).clamp_min(1e-12))>=.5).any():
        return "non_base_full"
    if ((inter/area.clamp_min(1e-12)>=.8)&(area<=.5*ref)).any():
        return "part_like"
    return "other"


def cohort(trace, rows):
    extra=json.loads((RELABEL/"scope_boxes.json").read_text())
    pos_ids=set(trace[trace[:,3]==1,0].astype(int))
    hard=trace[(trace[:,3]==0)&(trace[:,4]==1)]
    counts=Counter()
    picked={}
    quota={"part_like":64,"non_base_full":32}
    for t in hard[np.argsort(-hard[:,5],kind="stable")]:
        iid,q,c=map(int,t[:3])
        row=rows[iid]
        items=[x["bbox"] for x in row["annotations"] if not x["iscrowd"]]+extra[str(iid)]
        boxes=torch.tensor(items,dtype=torch.float32).reshape(-1,4)
        boxes[:,2:]+=boxes[:,:2]
        boxes/=torch.tensor([row["width"],row["height"]]*2)
        pack=load_pack(iid)
        group=subgroup(pack["boxes"][q].float(),boxes)
        counts[group]+=1
        if group in quota and quota[group] and iid not in pos_ids:
            quota[group]-=1
            picked[(iid,q,c)]=group
    assert len(pos_ids)==19 and counts=={"part_like":296,"non_base_full":88,"other":329},counts
    assert len(picked)==96 and all(v==0 for v in quota.values())
    return pos_ids,picked


def load_pack(iid):
    return torch.load(RELABEL/"cache/cal_policy"/(str(iid)+".pt"),map_location="cpu",weights_only=True)


class MaskEvidence:
    def __init__(self,path):
        self.sam=SamModel.from_pretrained(path,local_files_only=True,use_safetensors=True).cuda().eval()
        self.proc=SamProcessor.from_pretrained(path,local_files_only=True)

    def start(self,im):
        prep=self.proc(images=im,return_tensors="pt")
        self.h,self.w=map(int,prep["original_sizes"][0])
        h1,w1=map(int,prep["reshaped_input_sizes"][0])
        self.sizes=(prep["original_sizes"],prep["reshaped_input_sizes"])
        self.scale=torch.tensor([w1,h1],dtype=torch.float32,device="cuda")
        self.emb=self.sam.get_image_embeddings(prep["pixel_values"].cuda())
        self.x=torch.arange(self.w,device="cuda")[None,:]+.5
        self.y=torch.arange(self.h,device="cuda")[:,None]+.5

    def masks(self,boxes,points=None):
        args=dict(image_embeddings=self.emb,input_boxes=(boxes*self.scale.repeat(2))[None],multimask_output=True)
        if points is not None:
            args.update(input_points=(points*self.scale)[None,:,None],
                        input_labels=torch.ones((1,len(boxes),1),device="cuda",dtype=torch.long))
        pred=self.sam(**args)
        masks=self.proc.image_processor.post_process_masks(pred.pred_masks,*self.sizes)[0]
        assert masks.shape==(len(boxes),3,self.h,self.w)
        return masks,pred.iou_scores[0].float().clamp(0,1)

    def calculate(self,boxes):
        boxes=boxes.cuda().float()
        center=(boxes[:,:2]+boxes[:,2:])/2
        half=(boxes[:,2:]-boxes[:,:2])/2
        expanded=torch.cat((center-2*half,center+2*half),dim=1).clamp(0,1)
        t,q=self.masks(boxes)
        tight=t[torch.arange(len(t),device="cuda"),q.argmax(1)]
        x1,y1,x2,y2=(boxes*boxes.new_tensor([self.w,self.h,self.w,self.h])).unbind(1)
        rect=((self.x>=x1[:,None,None])&(self.x<x2[:,None,None])&
              (self.y>=y1[:,None,None])&(self.y<y2[:,None,None]))
        tight=tight&rect
        pixels=boxes.new_tensor([self.w,self.h])
        points=[]
        for mask,mid in zip(tight,center):
            yx=torch.nonzero(mask,as_tuple=False)
            if len(yx):
                xy=(yx[:,[1,0]].float()+.5)/pixels
                idx=((xy-mid)*pixels).square().sum(-1).argmin()
                points.append(xy[idx])
            else:
                points.append(mid)
        wide,q2=self.masks(expanded,torch.stack(points))
        inside=wide&rect[:,None]
        same=(inside&tight[:,None]).sum((2,3))/(inside|tight[:,None]).sum((2,3)).clamp_min(1)
        winner=(same*q2).argmax(1)
        result=[]
        for i,j in enumerate(winner.tolist()):
            mask=wide[i,j]
            yx=torch.nonzero(mask,as_tuple=False)
            if not len(yx) or not tight[i].any():
                result.append((0.,0.,0.))
                continue
            lo=yx.amin(0)[[1,0]].float()/pixels
            hi=(yx.amax(0)[[1,0]].float()+1)/pixels
            inter=(torch.minimum(hi,boxes[i,2:])-torch.maximum(lo,boxes[i,:2])).clamp_min(0).prod()
            b_iou=inter/((hi-lo).prod()+(boxes[i,2:]-boxes[i,:2]).prod()-inter).clamp_min(1e-8)
            spill=1-(mask&rect[i]).sum()/mask.sum()
            result.append((float(b_iou*same[i,j]),float(spill),float(same[i,j])))
        return np.asarray(result)


@torch.inference_mode()
def main():
    assert json.loads((RELABEL/"eval_deferred.json").read_text())["opened"] is False
    trace=np.load(RELABEL/"calibration_candidate_trace.npz")["rows"]
    rows={r["id"]:r for r in json.loads((RELABEL/"manifest.json").read_text())["splits"]["cal_policy"]}
    pos_ids,picked=cohort(trace,rows)
    selected_ids=pos_ids|{k[0] for k in picked}
    path=Path(os.environ["BASE"])/"models"/"sam-vit-base"
    receipt=json.loads((path/"extent_model_receipt.json").read_text())
    detector=MaskEvidence(path)
    outputs=[]
    for i,iid in enumerate(sorted(selected_ids),1):
        row=rows[iid]
        t=trace[trace[:,0]==iid]
        if iid not in pos_ids:
            t=t[[tuple(x[:3].astype(int)) in picked for x in t]]
        q,back=np.unique(t[:,1].astype(int),return_inverse=True)
        im=Image.open(RELABEL/"images"/row["dataset"]/row["file_name"]).convert("RGB")
        detector.start(im)
        p=load_pack(iid)["boxes"][torch.from_numpy(q)].float()
        measures=np.concatenate([detector.calculate(b) for b in p.split(8)])[back]
        labels=np.array([1 if x[3]==1 else
                         2 if picked.get(tuple(x[:3].astype(int)))=="part_like" else
                         3 if picked.get(tuple(x[:3].astype(int)))=="non_base_full" else 0 for x in t])
        outputs.append(np.column_stack((t,measures,labels)))
        if i%20==0 or i==len(selected_ids):
            print("extent",i,len(selected_ids),flush=True)
    out=np.concatenate(outputs)
    assert np.isfinite(out).all()
    np.savez_compressed("extent_predictions.npz",rows=out,
        columns=np.array(("image","query","class","tag","image_rank","utility","p_gain","p_spoil","p_align",
                          "extent","spill","inside_agreement","probe_group")))
    categories=out[:,12].astype(int)
    assert np.bincount(categories,minlength=4)[1:].tolist()==[20,64,32]
    median=lambda a:dict(extent=float(np.median(a[:,9])),spill=float(np.median(a[:,10])),
                          inside_agreement=float(np.median(a[:,11])))
    part=categories==2
    strict=categories==1
    keep=part|strict
    score=out[:,5]*out[:,9]
    top0=top1=0
    for iid in pos_ids:
        mask=out[:,0]==iid
        top0+=int(strict[mask][out[mask,5].argmax()])
        top1+=int(strict[mask][score[mask].argmax()])
    area=auc(strict[keep],out[keep,9])
    result=dict(images=len(selected_ids),model=receipt["repo"],revision=receipt["revision"],
        strict=median(out[strict]),part_like=median(out[part]),non_base_full=median(out[categories==3]),
        separation_auroc=area,strict_top1=dict(v6=top0,extent=top1,images=len(pos_ids)),
        joint_utility_hard_auroc=auc(strict[keep],score[keep]),passed=area>=.65 and top1>=6,
        source="opened v6 Cal targeted diagnostic")
    save_json("extent_probe.json",result)
    print(result,flush=True)


if __name__=="__main__":
    main()
