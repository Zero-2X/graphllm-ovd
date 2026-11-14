"""Frozen graph proposal features, annotated actions, and baseline-preserving edits."""
from pathlib import Path
import sys

import torch
from torch import nn
from torchvision.ops import batched_nms, box_iou

V1 = Path(__file__).resolve().parent.parent / "hierarchical_instance_v1"
sys.path.insert(0,str(V1))
from cache import vocabulary, gt_arrays
from model import StarAdapter

CAP=100
FLOOR=.05
NMS=.50
NEITHER,GAIN,HARM=0,1,2


def make_base(pack,nfine):
    boxes=pack["boxes"].float()
    scores=pack["logits"][:,:nfine].float().sigmoid()
    valid=(boxes[:,2:]>boxes[:,:2]).all(-1)
    q,c=torch.nonzero((scores>=FLOOR)&valid[:,None],as_tuple=True)
    values=scores[q,c]
    ix=batched_nms(boxes[q],values,c,NMS)[:CAP]
    return dict(query=q[ix],label=c[ix],score=values[ix],box=boxes[q[ix]])


def proposals(pack,base,nfine):
    r,e=torch.nonzero(pack.get("valid",torch.zeros((0,4),dtype=torch.bool)),as_tuple=True)
    if not len(r):
        return {}
    label=pack["labels"][r,e]
    keep=(label<nfine)&~pack["high"][r]
    r,e,label=r[keep],e[keep],label[keep]
    if not len(r):
        return {}
    q=pack["ids"][r]
    owner=torch.full((len(pack["boxes"]),),-1,dtype=torch.long)
    for j,x in enumerate(base["query"].tolist()):
        if owner[x]<0:
            owner[x]=j
    old=owner[q]
    old_label=pack["logits"][q,:nfine].argmax(-1)
    if len(base["label"]):
        old_label=torch.where(old>=0,base["label"][old.clamp_min(0)],old_label)
    eligible=(old<0)|(old_label!=label)
    if len(base["query"])>=CAP:
        eligible&=old>=0
    duplicate=(box_iou(pack["boxes"][q].float(),base["box"])>NMS)&(label[:,None]==base["label"][None])
    eligible&=~duplicate.any(-1)
    k=torch.nonzero(eligible).flatten()
    keys=set()
    unique=[]
    for a in k.tolist():
        key=(int(r[a]),int(label[a]))
        if key not in keys:
            unique.append(a)
            keys.add(key)
    if not unique:
        return {}
    k=torch.tensor(unique)
    return dict(receiver=r[k],edge=e[k],query=q[k],label=label[k],old=old[k],old_label=old_label[k],
                box=pack["boxes"][q[k]].float())


def training_targets(row,manifest,names,base,prop):
    truth,y=gt_arrays(row,names,manifest)
    covered=match(base,truth,y)
    old=prop["old"]
    n=len(old)
    closest=torch.full((n,),-1,dtype=torch.long)
    valid=torch.zeros(n,dtype=torch.bool)
    if len(truth):
        overlap,which=box_iou(prop["box"],truth).max(-1)
        closest=torch.where(overlap>=.50,which,-1)
        valid=(closest>=0)&(y[which]==prop["label"])&~torch.isin(which,covered[covered>=0])
    previously=torch.zeros(n,dtype=torch.bool)
    if len(covered):
        previously=(old>=0)&(covered[old.clamp_min(0)]>=0)
    # A replacement is evaluated as an edit to its receiver instance.
    target=torch.where(previously,HARM,torch.where(valid,GAIN,NEITHER))
    good=torch.ones(n,dtype=torch.bool)
    ids={manifest["categories"][x] for x in names if x in manifest["categories"]}
    crowd=torch.tensor([a["bbox"] for a in row["annotations"] if a["iscrowd"] and a["category_id"] in ids],
                       dtype=torch.float32).reshape(-1,4)
    if len(crowd):
        crowd[:,2:]+=crowd[:,:2]
        crowd/=torch.tensor([row["width"],row["height"],row["width"],row["height"]])
        bb=prop["box"]
        corners=torch.minimum(bb[:,None,2:],crowd[None,:,2:])-torch.maximum(bb[:,None,:2],crowd[None,:,:2])
        ratio=corners.clamp_min(0).prod(-1)/(bb[:,2:]-bb[:,:2]).prod(-1).clamp_min(1e-8)[:,None]
        good&=~((ratio.amax(-1)>.5)&(closest<0))
    return target,good


def match(output,truth,y):
    """Frozen IoU50 matching in score order for instance-level edit accounting."""
    match_ids=torch.full_like(output["label"],-1)
    if not len(y) or not len(match_ids):
        return match_ids
    overlaps=box_iou(output["box"],truth)
    overlaps.masked_fill_(output["label"][:,None]!=y[None],-1)
    assigned=torch.zeros(len(y),dtype=torch.bool)
    for i in output["score"].argsort(descending=True,stable=True).tolist():
        value,index=overlaps[i].masked_fill(assigned,-1).max(-1)
        if value>=.50:
            match_ids[i]=index
            assigned[index]=True
    return match_ids


class FrozenStar:
    def __init__(self):
        self.star=StarAdapter().cuda().eval()
        state=torch.load(V1/"best.pt",map_location="cpu",weights_only=True)
        missing,extra=self.star.load_state_dict(state,strict=False)
        assert not extra and all(x.startswith("qwen.") for x in missing)
        self.star.requires_grad_(False)

    def set_vocabulary(self,names,nfine):
        self.star.set_vocabulary(names,nfine,"cuda")

    @torch.inference_mode()
    def encode(self,pack,prop):
        star=self.star
        r=prop["receiver"]
        q=prop["query"]
        e=prop["edge"]
        raw=pack["logits"][q,:star.nfine].float()
        current=raw[torch.arange(len(q)),prop["old_label"]]
        proposed=raw[torch.arange(len(q)),prop["label"]]
        top=raw.topk(2,dim=-1).values
        box=prop["box"]
        wh=(box[:,2:]-box[:,:2]).clamp_min(1e-6)
        prior=torch.stack((current.sigmoid(),proposed.sigmoid(),(proposed-current).tanh(),
                           (top[:,0]-top[:,1]).tanh(),(prop["old"]>=0).float(),
                           wh.prod(-1).sqrt(),(wh[:,0]/wh[:,1]).log().tanh()),-1)
        hidden=[]
        distinct,inverse=torch.unique(r,sorted=True,return_inverse=True)
        for rr in distinct.split(96):
            valid=pack["valid"][rr].cuda()
            receiver=pack["features"][pack["i"][rr]].cuda().float()
            donor=pack["features"][pack["j"][rr]].cuda().float()
            emb=star.texts[pack["labels"][rr].cuda()]
            edge=star.edge(pack["stats"][rr].cuda().float())*star.embedding_scale
            projected=star.instance(donor)*star.embedding_scale
            self_feature=star.instance(receiver)*star.embedding_scale
            x=torch.cat((self_feature[:,None],projected+emb+edge,star.read[None,None]+emb+edge),1)
            mask=torch.cat((valid.new_ones((len(valid),1)),valid,valid),1).long()
            h=star.qwen(inputs_embeds=x.to(star.qwen.dtype),attention_mask=mask,
                        use_cache=False).last_hidden_state[:,-valid.shape[1]:].half().cpu()
            hidden.append(h)
        context=torch.cat(hidden)[inverse,e]
        return torch.cat((context,pack["stats"][r,e],prior.half()),-1)


class AdoptionHead(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.readout=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,128),nn.GELU(),nn.Linear(128,3))

    def forward(self,x,old):
        z=self.readout(x.float())
        z=z.masked_fill((old<0)[:,None] & (torch.arange(3,device=z.device)[None]==HARM),-1e4)
        return z


def prepare_image(row,manifest,names,nfine,encoder,path):
    pack=torch.load(path,map_location="cpu",weights_only=True)
    base=make_base(pack,nfine)
    prop=proposals(pack,base,nfine)
    x=encoder.encode(pack,prop) if prop else None
    return base,prop,x


def utility(probs):
    return probs[:,GAIN]-4*probs[:,HARM]-probs[:,NEITHER]


def winners(prop,probs):
    values=utility(probs)
    order=values.argsort(descending=True,stable=True)
    seen=set()
    out=[]
    for j in order.tolist():
        r=int(prop["receiver"][j])
        if r not in seen:
            out.append(j)
            seen.add(r)
    return out


def apply_edits(base,prop,probs,new_scores,cutoff):
    """No other native detection can be evicted by an edit."""
    if not prop or cutoff is None:
        return dict(base,origin=torch.arange(len(base["label"])),edit=torch.full_like(base["label"],-1))
    keep=torch.ones(len(base["label"]),dtype=torch.bool)
    adopted=[]
    values=utility(probs)
    overlap=box_iou(prop["box"],base["box"])
    same=prop["label"][:,None]==base["label"][None]
    for j in winners(prop,probs):
        if values[j]<=0 or values[j]<cutoff or new_scores[j]<FLOOR:
            continue
        old=int(prop["old"][j])
        if old>=0 and not keep[old]:
            continue
        if old<0 and int(keep.sum())+len(adopted)>=CAP:
            continue
        if ((overlap[j]>NMS)&same[j]&keep).any():
            continue
        if adopted:
            prev=torch.tensor(adopted)
            dup=(box_iou(prop["box"][j:j+1],prop["box"][prev])[0]>NMS)&(prop["label"][j]==prop["label"][prev])
            if dup.any():
                continue
        if old>=0:
            keep[old]=False
        adopted.append(j)
    j=torch.tensor(adopted,dtype=torch.long)
    assert int(keep.sum())+len(j)<=CAP
    result={k:torch.cat((base[k][keep],new_scores[j] if k=="score" else prop[k][j])) for k in base}
    result["origin"]=torch.cat((torch.nonzero(keep).flatten(),torch.full((len(j),),-1,dtype=torch.long)))
    result["edit"]=torch.cat((torch.full((int(keep.sum()),),-1,dtype=torch.long),j))
    return result


def count_edits(base,result,truth,labels):
    before=match(base,truth,labels)
    after=match(result,truth,labels)
    edited=result["edit"]>=0
    replaced=~torch.isin(torch.arange(len(before)),result["origin"])
    covered=torch.unique(before[before>=0])
    after_covered=torch.unique(after[after>=0])
    novel=edited&(after>=0)&~torch.isin(after,covered)
    return dict(edits=int(edited.sum()),valid_edits=int(novel.sum()),
                invalid_edits=int((edited&~novel).sum()),old_true=int(len(covered)),
                correct_lost=int((~torch.isin(covered,after_covered)).sum()),
                new_instances=int((~torch.isin(after_covered,covered)).sum()),
                original_correct_replaced=int((replaced&(before>=0)).sum()))
