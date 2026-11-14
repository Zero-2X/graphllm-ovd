"""One receiver-centered typed star, with frozen Qwen and separate residual heads."""
import os
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from torchvision.ops import generalized_box_iou_loss


class StarAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        path=Path(os.environ["BASE"])/"models/qwen2.5-0.5b"
        pretrained=AutoModelForCausalLM.from_pretrained(
            path,torch_dtype=torch.bfloat16,attn_implementation="sdpa",local_files_only=True)
        self.qwen=pretrained.model.requires_grad_(False).eval()
        self.tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
        d=self.qwen.config.hidden_size
        self.instance=nn.Sequential(nn.LayerNorm(768),nn.Linear(768,d))
        self.edge=nn.Linear(8,d)
        self.read=nn.Parameter(torch.zeros(d))
        self.edge_heads=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,128),nn.GELU(),nn.Linear(128,3))
        self.region=nn.Sequential(nn.LayerNorm(256),nn.Linear(256,8),nn.GELU())
        self.box_head=nn.Sequential(nn.Linear(25*9,96),nn.GELU(),nn.Linear(96,4))
        nn.init.zeros_(self.edge_heads[-1].weight)
        nn.init.zeros_(self.edge_heads[-1].bias)
        with torch.no_grad():
            self.edge_heads[-1].bias[1:].fill_(.50)
        nn.init.zeros_(self.box_head[-1].weight)
        nn.init.zeros_(self.box_head[-1].bias)
        self.register_buffer("texts",torch.empty(0,d),persistent=False)
        self.embedding_scale=float(self.qwen.embed_tokens.weight.float().std())

    def train(self,mode=True):
        super().train(mode)
        self.qwen.eval()
        return self

    @torch.no_grad()
    def set_vocabulary(self,names,nfine,device):
        self.nfine=nfine
        tokens=self.tokenizer([("specific class: " if i<nfine else "parent category: ")+c
                               for i,c in enumerate(names)],padding=True,return_tensors="pt").to(device)
        embedding=self.qwen.embed_tokens(tokens.input_ids).float()
        self.texts=(embedding*tokens.attention_mask[...,None]).sum(1)/tokens.attention_mask.sum(1,keepdim=True)

    def forward(self,batch,disabled=False):
        logits,boxes=batch["base"],batch["box"]
        if disabled or not batch["valid"].any():
            return logits,boxes
        mask=batch["valid"]
        k=mask.shape[1]
        s=self.instance(batch["self"].float())*self.embedding_scale
        projected=self.instance(batch["donor"].float())*self.embedding_scale
        label=self.texts[batch["labels"]]
        edge=self.edge(batch["stats"].float())*self.embedding_scale
        donor=projected+label+edge
        read=self.read[None,None,:]+label+edge
        inputs=torch.cat((s[:,None,:],donor,read),dim=1).to(self.qwen.dtype)
        attn=torch.cat((mask.new_ones((len(mask),1)),mask,mask),dim=1).long()
        hidden=self.qwen(inputs_embeds=inputs,attention_mask=attn,use_cache=False).last_hidden_state[:,-k:].float()
        raw,sem,geo=self.edge_heads(hidden).unbind(-1)
        semantic=mask & ~batch["high"][:,None]
        response=1.8*raw.tanh()*sem.tanh().relu()*semantic
        count=torch.zeros_like(logits).scatter_add(1,batch["labels"],semantic.float())
        updated=logits+torch.zeros_like(logits).scatter_add(1,batch["labels"],response)/count.clamp_min(1)
        # Parent evidence has no route to the separate concrete-class score columns.
        weights=geo.tanh().relu()*batch["loc"]
        map_=(batch["maps"].float()*weights[...,None]).sum(1)/weights.sum(1,keepdim=True).clamp_min(1)
        region=self.region(batch["grid"].float())
        inp=torch.cat((region,map_[...,None]),-1).flatten(1)
        own=torch.cat((region,torch.zeros_like(map_)[...,None]),-1).flatten(1)
        delta=(self.box_head(inp)-self.box_head(own)).tanh()
        wh=(boxes[:,2:]-boxes[:,:2]).clamp_min(1e-6)
        center=(boxes[:,:2]+boxes[:,2:])/2+.25*delta[:,:2]*wh
        size=wh*(.45*delta[:,2:]).exp()
        candidate=torch.cat((center-size/2,center+size/2),dim=-1).clamp(0,1)
        identity=boxes+(candidate-candidate.detach())
        refined=torch.where((delta!=0).any(-1,keepdim=True),candidate,identity)
        return updated,refined

    def adapter_state(self):
        return {k:v.detach().cpu() for k,v in self.state_dict().items() if not k.startswith("qwen.")}


def receiver_batch(pack,rows):
    ids=pack["ids"][rows]
    return {
        "base":pack["logits"][ids].float(),"box":pack["boxes"][ids].float(),
        "self":pack["features"][pack["i"][rows]],"donor":pack["features"][pack["j"][rows]],
        **{k:pack[k][rows] for k in ("valid","labels","loc","stats","high","grid","maps")},
    }


def per_row_loss(batch,updated,boxes,nfine):
    target=batch["y"]
    fine=F.cross_entropy(updated[:,:nfine],target,reduction="none")
    coarse=updated[:,nfine:]
    which=torch.arange(nfine,updated.shape[-1],device=updated.device)
    target_parent=(batch["parent"][:,None]==which).float()
    affected=torch.zeros_like(updated).scatter_add(1,batch["labels"],batch["valid"].float())[:,nfine:]>0
    p=F.binary_cross_entropy_with_logits(coarse,target_parent,reduction="none")
    parent=(p*affected).sum(-1)/affected.sum(-1).clamp_min(1)
    loc=F.l1_loss(boxes,batch["gt"],reduction="none").sum(-1)+generalized_box_iou_loss(boxes,batch["gt"])
    return fine+.45*parent+loc
