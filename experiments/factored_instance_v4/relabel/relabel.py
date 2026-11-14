"""Evidence and decision rule for replacing one existing weak class at its score."""
from pathlib import Path
import sys

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"pooled"))
from factor import FactorEvidence, V3, owner_slots, read_factor
from counterfactual import candidate_actions


def logodds(p):
    return p.clamp(.0001,.9999).logit()


@torch.inference_mode()
def summarize_evidence(model,grid,own,x,same):
    assert not same.any()
    device = next(model.parameters()).device
    grid,own,x,same = [a.to(device) for a in (grid,own,x,same)]
    quality = model.alignment_logits(grid,own).softmax(-1)
    z, donor = model.semantic_logits(x,same)
    p = z.softmax(-1)
    pair = x[:,model.context_dim:].float()
    # Graph statistics, then native decoder priors, then receiver CLIP class fit.
    fit = torch.stack((logodds(p[:,1]),logodds(p[:,2]),donor.clamp(-12,12),
        logodds(pair[:,8]),logodds(pair[:,9]),pair[:,15],pair[:,16],pair[:,19],
        pair[:,0],logodds(pair[:,2]),logodds(quality[:,2]/quality[:,1:].sum(-1).clamp_min(1e-8))),dim=-1)
    assert torch.isfinite(fit).all()
    return fit.cpu(),quality[:,1:].sum(-1).cpu()


class RelabelEvidence(FactorEvidence):
    @torch.inference_mode()
    def relabel(self,pack,base,model,allow=None):
        actions=[a for a in candidate_actions(pack,base,self.nfine) if a[0]=="replace"
                 and (allow is None or a[1:3] in allow)]
        if not actions:
            return actions,None,None
        r,e=torch.nonzero(pack["valid"]&(pack["labels"]<self.nfine)&~pack["high"][:,None],as_tuple=True)
        first={}
        for i,j in zip(r.tolist(),e.tolist()):
            first.setdefault((int(pack["ids"][i]),int(pack["labels"][i,j])),(i,j))
        r,e=torch.tensor([first[q,c] for _,q,c,_ in actions]).unbind(-1)
        x,same=self.claims(pack,base,r,e,owner_slots(pack,base))
        grid,own=self.spatial(pack,r)
        fit,align=summarize_evidence(model,grid,own,x,same)
        return actions,fit,align


class RelabelGate(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.readout=nn.Linear(width,3)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)
        self.register_buffer("mean",torch.zeros(width))
        self.register_buffer("scale",torch.ones(width))
        self.register_buffer("weight",torch.ones(3))

    def forward(self,x):
        return self.readout(((x-self.mean)/self.scale).clamp(-6,6))

    def posterior(self,x):
        return (self(x)-self.weight.log()).softmax(-1)

    @torch.inference_mode()
    def rank(self,fit,alignment):
        p=self.posterior(fit)
        gain=alignment*p[:,1]
        spoil=alignment*p[:,2]
        utility=gain/(1-gain+4*spoil).clamp_min(1e-6)
        return utility,torch.stack((gain,spoil,alignment),dim=-1)


def load_gate(path="best.pt"):
    state=torch.load(path,map_location="cpu",weights_only=True)
    model=RelabelGate(state["width"])
    model.load_state_dict(state["model"])
    return model.eval()
