"""Frozen Qwen, matched real/NULL reads, and receiver-only visual values."""
import math
import os
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .native_bridge import grid_xy


def center(x):
    return x-x.mean(-1,keepdim=True)


def normalize(x,floor,clip,mean,std):
    return ((x.float()-mean)/std.clamp_min(floor)).clamp(-clip,clip)


class Receiver(nn.Module):
    def __init__(self,cfg,scale):
        super().__init__()
        d=cfg["dimensions"]
        h=d["hidden"]
        self.global_=nn.Sequential(nn.LayerNorm(d["node"]),nn.Linear(d["node"],h))
        self.local=nn.Sequential(nn.LayerNorm(d["local"]),nn.Linear(d["local"],h))
        self.native=nn.Linear(d["native"],h)
        self.position=nn.Linear(2,h,bias=False)
        self.role=nn.Parameter(torch.randn(5,h)*scale)
        self.base=nn.Parameter(torch.randn(h)*scale)
        self.head=nn.Sequential(nn.LayerNorm(h),nn.Linear(h,d["base_mlp"]),nn.GELU(),nn.Linear(d["base_mlp"],3))
        self.register_buffer("mean",torch.zeros(d["native"]))
        self.register_buffer("std",torch.ones(d["native"]))
        self.scale=scale
        self.cfg=cfg

    def forward(self,b,text,xy):
        phi=F.layer_norm(self.local(b["local"].float()).float(),(len(self.base),))
        native=normalize(b["native"],self.cfg["normalizer"]["floor"],self.cfg["normalizer"]["clip"],self.mean,self.std)
        first=text+self.role[:2]
        glob=self.scale*F.layer_norm(self.global_(b["receiver"].float()).float(),(len(self.base),))+self.role[2]
        state=self.scale*F.layer_norm(self.native(native).float(),(len(self.base),))+self.role[3]
        local=self.scale*(phi+self.position(xy).float())+self.role[4]
        prefix=torch.cat((first,glob[:,None],state[:,None],local,self.base.expand(len(phi),1,-1)),1)
        return prefix,phi


class Evidence(nn.Module):
    def __init__(self,cfg,receiver,scale):
        super().__init__()
        from copy import deepcopy
        d=cfg["dimensions"]
        h,m=d["hidden"],d["message"]
        self.global_=deepcopy(receiver.global_).requires_grad_(True)
        self.local=deepcopy(receiver.local).requires_grad_(True)
        self.position=deepcopy(receiver.position).requires_grad_(True)
        self.edge=nn.Linear(d["edge"],h)
        self.role=nn.Parameter(torch.randn(3,h)*scale)
        self.read=nn.Parameter(torch.randn(h)*scale)
        self.relation=nn.Sequential(nn.LayerNorm(h),nn.Linear(h,d["relation_mlp"]),nn.GELU(),nn.Linear(d["relation_mlp"],3))
        self.query=nn.Linear(h,m,bias=False)
        self.key=nn.Linear(h,m,bias=False)
        self.value=nn.Linear(h,m,bias=False)
        self.key_pos=nn.Linear(2,m,bias=False)
        self.null=nn.Parameter(torch.randn(m)/math.sqrt(m))
        self.null_bias=nn.Parameter(torch.tensor(math.log(d["grid"]**2)))
        self.wm=nn.Linear(m,m,bias=False)
        self.wo=nn.Linear(m,3,bias=False)
        self.wb=nn.Linear(h,m)
        nn.init.zeros_(self.wo.weight)
        self.scale=scale

    def suffix(self,glob,local,edge,label,xy):
        h=len(self.read)
        g=self.scale*F.layer_norm(self.global_(glob.float()).float(),(h,))+label+self.role[0]
        p=F.layer_norm(self.local(local.float()).float(),(h,))
        p=self.scale*(p+self.position(xy).float())+self.role[1]
        e=self.scale*F.layer_norm(self.edge(edge.float()).float(),(h,))+self.role[2]
        return torch.cat((g[:,None],p,e[:,None],self.read.expand(len(g),1,-1)),1)

    def correct(self,message,base):
        return center(self.wo(F.gelu(self.wm(message))*torch.sigmoid(self.wb(F.layer_norm(base,(base.shape[-1],))))))

    def apply(self,base,hidden,raw_mask,xy,diagnose=False):
        # Slot zero is NULL for every row, later slots are physical donors.
        q=self.query(F.layer_norm(hidden.float(),(hidden.shape[-1],))).float()
        k=self.key(base["phi"].float()).float()+self.key_pos(xy).float()
        v=self.value(base["phi"].float()).float()
        inv=1/math.sqrt(q.shape[-1])
        scores=(q@k.transpose(-1,-2))*inv
        beta=scores.softmax(-1)
        # Difference of patch distributions; identical visual values cancel exactly.
        m=(beta[:,1:]-beta[:,:1])@(v-v[:,:1])
        rel=self.relation(hidden[:,1:].float()).float()
        prob=rel.softmax(-1)
        s=(scores[:,1:].logsumexp(-1)-(q[:,1:]@self.null)*inv-self.null_bias).sigmoid()
        g=s*(1-prob[...,2].detach())*raw_mask
        t=g[...,None]*m
        den=raw_mask.sum(1).clamp_min(1)
        msg=t.sum(1)/den[:,None]
        delta=self.correct(msg,base["hidden"]).float()
        out=dict(base=base["z"],final=base["z"]+delta,message=msg,delta=delta,t=t,
                 relation=rel,relation_probs=prob,nonnull=s,gate=g)
        if diagnose:
            z=base["z"][:,None]+self.correct(t,base["hidden"][:,None])
            p=z.float().softmax(-1)
            ent=lambda a: -(a*a.clamp_min(1e-12).log()).sum(-1)
            w=raw_mask/den[:,None]
            js=(ent((p*w[...,None]).sum(1))-(w*ent(p)).sum(1)).clamp_min(0)
            out.update(js=js,js_available=den>=2)
        return out


class GraphReader(nn.Module):
    def __init__(self,doc,cfg,device):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer

        path=Path(os.environ["BASE"])/"models/qwen2.5-0.5b"
        dtype=torch.bfloat16 if torch.device(device).type=="cuda" and torch.cuda.is_bf16_supported() else torch.float32
        self.qwen=AutoModelForCausalLM.from_pretrained(path,torch_dtype=dtype,attn_implementation="sdpa",
                  local_files_only=True).model.requires_grad_(False).eval().to(device)
        assert self.qwen.config.hidden_size==cfg["dimensions"]["hidden"]
        tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
        ids=tokenizer([f"specific class: {c}" for c in doc["base"]],padding=True,return_tensors="pt").to(device)
        with torch.no_grad():
            embedding=self.qwen.embed_tokens(ids.input_ids).float()
            self.register_buffer("texts",(embedding*ids.attention_mask[...,None]).sum(1)/
                                 ids.attention_mask.sum(1,keepdim=True),persistent=False)
        self.register_buffer("xy",grid_xy(cfg["dimensions"]["grid"],device),persistent=False)
        self.scale=float(self.qwen.embed_tokens.weight.detach().float().std())
        self.receiver=Receiver(cfg,self.scale).to(device)
        self.cfg=cfg

    def train(self,mode=True):
        super().train(mode)
        self.qwen.eval()
        if hasattr(self,"evidence"):
            self.receiver.eval()
        return self

    def add_evidence(self):
        assert not hasattr(self,"evidence")
        self.receiver.requires_grad_(False).eval()
        self.evidence=Evidence(self.cfg,self.receiver,self.scale).to(self.texts.device)

    def read(self,seq):
        seq=seq.to(self.qwen.dtype)
        n=seq.shape[1]
        d=self.cfg["dimensions"]
        prefix=d["grid"]**2+5
        length=2*(d["grid"]**2)+8
        # Use a common attention length so the BF16 prefix has the same kernel in both reads.
        # Synthetic padding is future-masked and never reaches the BASE or REL readout.
        assert n in (prefix,length)
        if n<length:
            seq=F.pad(seq,(0,0,0,length-n))
        pos=torch.arange(length,device=seq.device).expand(len(seq),-1)
        future=pos[0,:,None]<pos[0,None,:]
        mask=seq.new_zeros(1,1,length,length).masked_fill(future,torch.finfo(seq.dtype).min)
        hidden=self.qwen(inputs_embeds=seq,attention_mask=mask.expand(len(seq),-1,-1,-1),position_ids=pos,
                         use_cache=False,output_hidden_states=False).last_hidden_state
        return hidden[:,:n].float()

    def read_base(self,b):
        text=self.texts[b["classes"]]
        prefix,phi=self.receiver(b,text,self.xy)
        hidden=self.read(prefix)[:,-1]
        return dict(prefix=prefix,phi=phi,hidden=hidden,z=self.receiver.head(hidden).float(),label=text[:,1])

    def read_relation(self,b,base,all_positions=False):
        raw=b["edge_mask"]
        valid=torch.cat((raw.any(1,keepdim=True),raw),1)
        i,j=valid.nonzero(as_tuple=True)
        def with_zero(x):
            return torch.cat((torch.zeros_like(x[:,:1]),x),1)[i,j]
        suffix=self.evidence.suffix(with_zero(b["donor"]),with_zero(b["donor_local"]),
                                    with_zero(b["edge"]),base["label"][i],self.xy)
        seq=torch.cat((base["prefix"][i],suffix),1)
        read=self.read(seq)
        if all_positions:
            return read,i,j
        dense=read.new_zeros((*valid.shape,read.shape[-1]))
        return dense.index_put((i,j),read[:,-1])

    def forward(self,b,stage="message",off=False,diagnose=False,base=None):
        if base is None:
            if stage=="self":
                return self.read_base(b)["z"]
            with torch.no_grad():
                base=self.read_base(b)
        raw=b["edge_mask"]
        if off or not raw.any():
            z=base["z"]
            return dict(base=z,final=z,message=z.new_zeros(len(z),self.cfg["dimensions"]["message"]),
                        delta=torch.zeros_like(z))
        hidden=self.read_relation(b,base)
        return self.evidence.apply(base,hidden,raw,self.xy,diagnose)


class Gain(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        d=cfg["dimensions"]
        self.register_buffer("mean",torch.zeros(d["gain"]))
        self.register_buffer("std",torch.ones(d["gain"]))
        self.net=nn.Sequential(nn.Linear(d["gain"],d["gain_mlp"]),nn.GELU(),nn.Linear(d["gain_mlp"],1))
        nn.init.zeros_(self.net[-1].bias)
        self.cfg=cfg

    def forward(self,x):
        settings=self.cfg["normalizer"]
        return self.net(normalize(x,settings["floor"],settings["clip"],self.mean,self.std)).flatten()


def gain_input(native16,risk,z,msg):
    return torch.cat((native16.float(),risk.float()[:,None],(z[:,1]-z[:,0])[:,None],
                      z.float().softmax(-1)[:,2:3],msg.float()),1)
