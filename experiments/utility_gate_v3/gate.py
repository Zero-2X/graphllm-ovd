"""One claim-pair readout using frozen local and same-image donor evidence."""
import os
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoProcessor,CLIPModel

from counterfactual import PROTECT,candidate_actions
from paired_targets import KINDS,LEVELS
from selective import CAP,FrozenStar


class FrozenEvidence:
    def __init__(self):
        self.graph=FrozenStar()
        path=Path(os.environ["BASE"])/"models"/"clip"
        self.clip=CLIPModel.from_pretrained(path,local_files_only=True,torch_dtype=torch.float16,
                                            use_safetensors=True).cuda().eval().requires_grad_(False)
        self.processor=AutoProcessor.from_pretrained(path,local_files_only=True)
        self.width=self.graph.star.qwen.config.hidden_size

    @torch.inference_mode()
    def vocabulary(self,names,nfine,parents):
        self.nfine=nfine
        self.parents=torch.tensor(parents)
        self.graph.set_vocabulary(names,nfine)
        txt=self.processor(text=[f"a photo of a {n}." for n in names[:nfine]],padding=True,
                           return_tensors="pt").to("cuda")
        self.text=F.normalize(self.clip.get_text_features(**txt).float(),dim=-1).cpu()

    @torch.inference_mode()
    def encode(self,pack,base):
        actions=[(a,score) for score in LEVELS for a in candidate_actions(pack,base,self.nfine,score)]
        if not actions:
            return actions,None
        by_claim={}
        r,e=torch.nonzero(pack["valid"]&(pack["labels"]<self.nfine)&~pack["high"][:,None],as_tuple=True)
        for i,j in zip(r.tolist(),e.tolist()):
            key=(int(pack["ids"][i]),int(pack["labels"][i,j]))
            by_claim.setdefault(key,(i,j))
        keys=list(dict.fromkeys((a[1],a[2]) for a,_ in actions))
        index={k:i for i,k in enumerate(keys)}
        q,c=torch.tensor(keys).unbind(-1)
        r,e=torch.tensor([by_claim[k] for k in keys]).unbind(-1)
        owner=torch.full((len(pack["boxes"]),),-1,dtype=torch.long)
        for i,j in enumerate(base["query"].tolist()):
            if owner[j]<0:
                owner[j]=i
        old=owner[q]
        old_label=pack["logits"][q,:self.nfine].argmax(-1)
        if len(base["label"]):
            old_label=torch.where(old>=0,base["label"][old.clamp_min(0)],old_label)
        prop=dict(receiver=r,edge=e,query=q,label=c,old=old,old_label=old_label,box=pack["boxes"][q])
        graph=self.graph.encode(pack,prop).float()
        fit=pack["features"][pack["i"][r],256:].float()@self.text.T
        step=torch.arange(len(q))
        old_fit=fit[step,old_label]
        new_fit=fit[step,c]
        sibling=(self.parents[None]==self.parents[c,None])&(self.parents[c,None]>=0)
        sibling[step,c]=False
        best=fit.amax(-1)
        other=fit.masked_fill(~sibling,-1).amax(-1)
        other=torch.where(sibling.any(-1),other,best)
        local=torch.stack((old_fit,new_fit,new_fit-old_fit,new_fit-other,new_fit-best,
                           ((self.parents[c]==self.parents[old_label])&(self.parents[c]>=0)).float(),(c!=old_label).float()),-1)
        base_feature=torch.cat((graph,local),-1)
        action_features=[]
        match=[]
        for (kind,q,c,slot),score in actions:
            own=int(owner[q])
            current=float(base["score"][own]) if own>=0 else 0.
            displaced=float(base["score"][slot]) if slot is not None else 0.
            action_features.append([*(float(kind==k) for k in KINDS),score,current,displaced,
                                    score-current,len(base["score"])/CAP,float(slot is not None)])
            match.append(index[q,c])
        x=torch.cat((base_feature[torch.tensor(match)],torch.tensor(action_features,dtype=torch.float32)),-1)
        assert torch.isfinite(x).all()
        return actions,x.half()


class PairGate(nn.Module):
    def __init__(self,dim,context_dim):
        super().__init__()
        self.context_dim=context_dim
        self.context=nn.LayerNorm(context_dim)
        self.readout=nn.Sequential(nn.Linear(dim,160),nn.GELU(),nn.Linear(160,7))

    def forward(self,x):
        x=x.float()
        context=self.context(x[:,:self.context_dim])
        return self.readout(torch.cat((context,x[:,self.context_dim:]*2),-1))

    def corrected(self,x,joint_weight,positive_weight):
        z=self(x)
        return z[:,:4]-joint_weight.log(),z[:,4:]-positive_weight.log()


def read_model(path="best.pt"):
    state=torch.load(path,map_location="cpu",weights_only=True)
    model=PairGate(state["dim"],state["context_dim"]).cuda().eval()
    model.load_state_dict(state["model"])
    return model,state["joint_weight"].cuda(),state["positive_weight"].cuda()


@torch.inference_mode()
def action_values(actions,x,base,head,joint_weight,positive_weight):
    a,s=head.corrected(x.cuda(),joint_weight,positive_weight)
    p=a.softmax(-1)
    sem=s.sigmoid()
    utility=p[:,1]-4*(p[:,2]+p[:,3])
    owners={}
    for q,c,score in zip(base["query"].tolist(),base["label"].tolist(),base["score"].tolist()):
        owners.setdefault(q,(c,score))
    change=torch.tensor([q in owners and c!=owners[q][0] and
                         (kind=="replace" or score>owners[q][1])
                         for (kind,q,c,_),score in actions],device=utility.device)
    return utility, (~change)|(sem[:,1]>sem[:,0]), sem


@torch.inference_mode()
def select_action(actions,x,base,head,joint_weight,positive_weight):
    if not actions:
        return None
    utility,allowed,sem=action_values(actions,x,base,head,joint_weight,positive_weight)
    if not allowed.any():
        return None
    choice=int(utility.masked_fill(~allowed,-torch.inf).argmax())
    return choice,float(utility[choice]),[float(v) for v in sem[choice]]
