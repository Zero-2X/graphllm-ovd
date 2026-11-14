"""Encode old and nominated classes with the same GraphLLM evidence contract."""
import json
from pathlib import Path
import sys

import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"relabel"))
from relabel import RelabelEvidence
from counterfactual import candidate_actions,make_base,save_json,vocabulary
from paired_targets import V1


@torch.inference_mode()
def hypotheses(reader,pack,queries,classes):
    """Return recipient evidence and donor residual for each class of each receiver."""
    model=reader.graph.star
    device=model.texts.device
    lookup={int(q):i for i,q in enumerate(pack["ids"])}
    rows=torch.tensor([lookup[int(q)] for q in queries]).repeat_interleave(classes.shape[1])
    c=classes.flatten()
    text=model.texts[c.to(device)]
    r=pack["features"][pack["i"][rows]].to(device).float()
    source=pack["features"][pack["j"][rows]].to(device).float()
    edge=pack["stats"][rows].to(device).float()
    valid=(pack["valid"][rows]&(pack["labels"][rows]==c[:,None])&~pack["high"][rows,None]).to(device)
    found=valid.any(-1)
    scale=model.embedding_scale
    self_feature=model.instance(r)*scale
    transfer=(model.instance(source)*scale+model.edge(edge)*scale+text[:,None])
    transfer=(transfer*valid[...,None]).sum(1)/valid.sum(1,keepdim=True).clamp_min(1)
    read=model.read[None]+text
    seq=torch.stack((self_feature,transfer,read),dim=1)
    present=torch.ones((len(seq),3),device=device,dtype=torch.long)
    absent=present.clone()
    present[:,1]=found
    absent[:,1]=0
    hs=model.qwen(inputs_embeds=torch.cat((seq,seq)).to(model.qwen.dtype),
                  attention_mask=torch.cat((present,absent)),
                  position_ids=torch.arange(3,device=device)[None].expand(2*len(seq),-1),
                  use_cache=False).last_hidden_state[:,-1].float()
    conditioned,own=hs.chunk(2)
    residual=torch.where(found[:,None],conditioned-own,0.)
    q=queries[:,None].expand_as(classes).flatten()
    native=pack["logits"][q,c].float()
    clip=(r[:,256:].cpu() * reader.text[c]).sum(-1)
    n,k=classes.shape
    return dict(own=own.reshape(n,k,-1).cpu(),donor_residual=residual.reshape(n,k,-1).cpu(),
                supported=found.reshape(n,k).cpu(),native=native.reshape(n,k),clip=clip.reshape(n,k))


def main():
    manifest=json.loads((V1/"manifest.json").read_text())
    names,nfine,parents=vocabulary(manifest,"cal")
    reader=RelabelEvidence()
    reader.vocabulary(names,nfine,parents)
    result=dict(images=0,pairs=0,old_with_support=0,nominated_with_support=0,
                absent_residual_max=0.,supported_residual_max=0.,swap_max=0.)
    for row in manifest["splits"]["cal"]:
        pack=torch.load(V1/"cache/cal"/(str(row["id"])+".pt"),map_location="cpu",weights_only=True)
        base=make_base(pack,nfine)
        actual=[a for a in candidate_actions(pack,base,nfine) if a[0]=="replace"][:12]
        if not actual:
            continue
        q=torch.tensor([a[1] for a in actual])
        pair=torch.tensor([[int(base["label"][slot]),c] for _,_,c,slot in actual])
        evidence=hypotheses(reader,pack,q,pair)
        swapped=hypotheses(reader,pack,q,pair.flip(1))
        support=evidence["supported"]
        assert support[:,1].all()
        for k,t in evidence.items():
            rev=swapped[k].flip(1)
            if t.dtype==torch.bool:
                assert torch.equal(rev,t)
            else:
                assert torch.isfinite(t).all()
                result["swap_max"]=max(result["swap_max"],float((rev-t).abs().max()))
        res=evidence["donor_residual"].abs()
        if (~support).any():
            result["absent_residual_max"]=max(result["absent_residual_max"],float(res[~support].max()))
        result["supported_residual_max"]=max(result["supported_residual_max"],float(res[support].max()))
        result["images"]+=1
        result["pairs"]+=len(q)
        result["old_with_support"]+=int(support[:,0].sum())
        result["nominated_with_support"]+=int(support[:,1].sum())
        if result["images"]>=8:
            break
    assert result["images"]==8 and result["pairs"]>=24
    assert result["old_with_support"]<result["pairs"]
    assert result["absent_residual_max"]==0 and result["supported_residual_max"]>0 and result["swap_max"]<=.035
    save_json("hypothesis_smoke.json",result)
    print(result,flush=True)


if __name__=="__main__":
    main()
