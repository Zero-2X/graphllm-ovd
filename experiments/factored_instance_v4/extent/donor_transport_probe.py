"""Verify whether actual same-image donor edges provide full, rather than partial, support."""
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torchvision.ops import box_iou, roi_align
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from extent_probe import RELABEL,auc,load_pack,save_json
from counterfactual import ImageReplay,make_base,vocabulary
from cache import xyxy

SIZE=7
EPS=.07
STEPS=40


def cohorts():
    extent=np.load("extent_predictions.npz")["rows"]
    trace=np.load(RELABEL/"calibration_candidate_trace.npz")["rows"]
    selected=[(*map(int,row[:3]),1 if int(row[12])==1 else 0)
              for row in extent if int(row[12]) in (1,2)]
    used={x[0] for x in selected}
    wrong=trace[trace[:,3]==2]
    for row in wrong[np.argsort(-wrong[:,5],kind="stable")]:
        iid,q,c=map(int,row[:3])
        if iid in used:
            continue
        selected.append((iid,q,c,2))
        used.add(iid)
        if len(selected)==148:
            break
    a=np.asarray(selected,dtype=np.int64)
    assert np.bincount(a[:,3],minlength=3).tolist()==[64,20,64]
    return a


class Native:
    def __init__(self,names):
        path=Path(os.environ["BASE"])/"models/dino"
        self.model=AutoModelForZeroShotObjectDetection.from_pretrained(path,local_files_only=True,
                    disable_custom_kernels=True).cuda().eval()
        self.proc=AutoProcessor.from_pretrained(path,local_files_only=True)
        self.prompt=". ".join(names)+"."
        t=self.proc.tokenizer(self.prompt,return_offsets_mapping=True)
        self.ids=t.input_ids
        cats=[]
        tok=[]
        offset=0
        for i,name in enumerate(names):
            end=offset+len(name)
            ids=[j for j,(a,b) in enumerate(t.offset_mapping) if a<b and offset<=a and b<=end]
            assert ids
            cats.extend([i]*len(ids))
            tok.extend(ids)
            offset=end+2
        self.cat=torch.tensor(cats,device="cuda")
        self.tok=torch.tensor(tok,device="cuda")
        self.n=len(names)
        self.capture={}
        def hook(_module,_args,kwargs):
            self.capture.update({k:kwargs[k] for k in ("vision_encoder_hidden_states","spatial_shapes","valid_ratios")})
        self.handle=self.model.model.decoder.register_forward_pre_hook(hook,with_kwargs=True)

    def forward(self,image):
        x=self.proc(images=image,text=self.prompt,return_tensors="pt").to("cuda")
        assert x.input_ids[0].tolist()==self.ids
        raw=self.model(**x)
        self.last_output=raw
        z=raw.logits[0]
        scores=z.new_full((len(z),self.n),-torch.inf)
        scores.scatter_reduce_(1,self.cat.expand(len(z),-1),z[:,self.tok],reduce="amax")
        h,w=self.capture["spatial_shapes"][0].tolist()
        latent=self.capture["vision_encoder_hidden_states"][0,:h*w].float().reshape(h,w,-1)
        ratio=self.capture["valid_ratios"][0,0]
        size=ratio*latent.new_tensor([w,h])
        nw,nh=[int(round(float(k))) for k in size]
        mean=latent[:nh,:nw].mean((0,1),keepdim=True)
        field=(latent-mean).permute(2,0,1).unsqueeze(0).contiguous()
        return xyxy(raw.pred_boxes[0]),scores,field,size.repeat(2)


def transport(sim):
    assert sim.ndim==3 and sim.shape[1:]==(SIZE*SIZE,SIZE*SIZE)
    kernel=sim/EPS
    marginal=-float(np.log(SIZE*SIZE))
    u=torch.zeros_like(kernel[:,:,0])
    v=u.clone()
    for _ in range(STEPS):
        u=marginal-torch.logsumexp(kernel+v[:,None,:],dim=2)
        v=marginal-torch.logsumexp(kernel+u[:,:,None],dim=1)
    plan=(kernel+u[:,:,None]+v[:,None,:]).exp()
    error=max(float((plan.sum(2)-np.exp(marginal)).abs().max()),
              float((plan.sum(1)-np.exp(marginal)).abs().max()))
    assert error<.005,error
    return (plan*sim).sum((1,2)),error


def actual_donors(pack,nfine):
    r,e=torch.nonzero(pack["valid"]&(pack["labels"]<nfine)&~pack["high"][:,None],as_tuple=True)
    first={}
    for i,j in zip(r.tolist(),e.tolist()):
        key=(int(pack["ids"][i]),int(pack["labels"][i,j]))
        first.setdefault(key,int(pack["node_ids"][pack["j"][i,j]]))
    return first


@torch.inference_mode()
def main():
    manifest=json.loads((RELABEL/"manifest.json").read_text())
    assert not json.loads((RELABEL/"eval_deferred.json").read_text())["opened"]
    names,nfine,_=vocabulary(manifest,"cal_policy")
    selected=cohorts()
    unique=set(selected[:,0].tolist())
    native=Native(names)
    rows=[]
    identity=None
    max_error=0.
    for row in manifest["splits"]["cal_policy"]:
        iid=row["id"]
        if iid not in unique:
            continue
        pack=load_pack(iid)
        replay=ImageReplay(row,pack,make_base(pack,nfine),manifest,names)
        actual=actual_donors(pack,nfine)
        keys=selected[selected[:,0]==iid]
        qs=keys[:,1]
        ds=[actual[q,c] for _,q,c,_ in keys]
        image=Image.open(RELABEL/"images"/row["dataset"]/row["file_name"]).convert("RGB")
        boxes,logits,field,scale=native.forward(image)
        if identity is None:
            identity=dict(box=float((boxes.cpu()-pack["boxes"]).abs().max()),
                          logits=float((logits.cpu()-pack["logits"]).abs().max()))
            assert identity["box"]<=1e-4 and identity["logits"]<=.002,identity
        patch=roi_align(field,[boxes[np.concatenate((qs,ds))]*scale],SIZE,sampling_ratio=2,aligned=True)
        x=F.normalize(patch.flatten(2).transpose(1,2),dim=-1)
        recv,donor=x.chunk(2)
        similarity=donor@recv.transpose(1,2)
        score,err=transport(similarity)
        max_error=max(max_error,err)
        values=torch.stack((score,similarity.max(2).values.mean(1),similarity.max(1).values.mean(1)),dim=1).cpu().numpy()
        for j,((_,q,c,kind),d) in enumerate(zip(keys,ds)):
            matched=False
            if len(replay.y):
                source=int(replay.iou[d].argmax())
                rec=int(replay.iou[q].argmax())
                matched=bool(replay.iou[d,source]>=.5 and replay.y[source]==c and source!=rec)
            rows.append((iid,q,c,kind,int(matched),*values[j]))
        if len(rows)%20==0 or len(rows)==len(selected):
            print("transport",len(rows),len(selected),flush=True)
    a=np.asarray(rows,dtype=np.float64)
    assert len(a)==148 and np.isfinite(a).all()
    pos=a[:,3]==1
    labels={"part_like":a[:,3]==0,"old_harmed":a[:,3]==2}
    scores={}
    for name,negative in labels.items():
        mask=pos|negative
        scores[name]=dict(transport=auc(pos[mask],a[mask,5]),donor_to_receiver_nearest=auc(pos[mask],a[mask,6]),
            receiver_to_donor_nearest=auc(pos[mask],a[mask,7]))
    result=dict(images=len(unique),rows=len(a),identity=identity,marginal_max_error=max_error,comparisons=scores,
        medians={name:dict(transport=float(np.median(a[mask,5])),true_independent_donor=int(a[mask,4].sum()))
                 for name,mask in dict(strict=pos,**labels).items()},
        passed=scores["part_like"]["transport"]>=.65 and scores["old_harmed"]["transport"]>=.60,
        source="already opened development diagnostic")
    np.savez_compressed("donor_transport_predictions.npz",rows=a,
        columns=np.array(("image","query","class","tag","true_donor","transport","donor_to_receiver","receiver_to_donor")))
    save_json("donor_transport_probe.json",result)
    print(result,flush=True)


if __name__=="__main__":
    main()
