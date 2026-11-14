"""Cache complete-object evidence and paired class scores for actual relabel actions."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from extent_probe import MaskEvidence, RELABEL
from paired_hypotheses import RelabelEvidence,V1,hypotheses,save_json,vocabulary
from train_hypothesis import SharedHypothesisHead
from relabel import read_factor
from counterfactual import make_base
from train_factor import by_image

HERE=Path(__file__).resolve().parent
MINING=HERE.parent/"mining"
POOLED=HERE.parent/"pooled"


class Evidence:
    def __init__(self):
        path=Path(os.environ["BASE"])/"models"/"sam-vit-base"
        assert (path/"extent_model_receipt.json").is_file()
        self.segment=MaskEvidence(path)
        self.reader=RelabelEvidence()
        self.geometry=read_factor(POOLED/"best.pt")
        checkpoint=torch.load(HERE/"hypothesis_best.pt",map_location="cpu",weights_only=True)
        self.classifier=SharedHypothesisHead(checkpoint["width"]).cuda().eval()
        self.classifier.load_state_dict(checkpoint["model"])

    @torch.inference_mode()
    def encode(self,image,pack,query,classes):
        keys,inverse=torch.unique(query,sorted=True,return_inverse=True)
        self.segment.start(image)
        measures=np.concatenate([self.segment.calculate(b) for b in pack["boxes"][keys].float().split(8)])
        graph={q:r for r,q in enumerate(pack["ids"].tolist())}
        r=torch.tensor([graph[q] for q in keys.tolist()],dtype=torch.long)
        grid,own=self.reader.spatial(pack,r)
        z=self.geometry.alignment_logits(grid.cuda(),own.cuda()).softmax(-1).cpu()
        spatial=torch.cat((z[:,1:].sum(-1,keepdim=True),z[:,2:],torch.from_numpy(measures).float()),dim=-1)
        pair=hypotheses(self.reader,pack,query,classes)
        assert pair["supported"][:,1].all()
        device={k:v.cuda() for k,v in pair.items()}
        pred=self.classifier(device,torch.arange(len(query),device="cuda")).softmax(-1).cpu()
        result=torch.cat((spatial[inverse],pred,pair["supported"][:,0:1].float()),dim=-1)
        assert torch.isfinite(result).all()
        return result.numpy()


def sources(name):
    if name=="mine":
        return MINING,"mine_train",MINING/"mined_targets.npz"
    return V1,name,POOLED/("relabel_"+name+".npz")


@torch.inference_mode()
def process(evidence,name,shard,shards):
    root,split,labels=sources(name)
    manifest=json.loads((root/"manifest.json").read_text())
    names,nfine,parents=vocabulary(manifest,split)
    evidence.reader.vocabulary(names,nfine,parents)
    with np.load(labels) as data:
        cols={str(k):i for i,k in enumerate(data["columns"])}
        targets=by_image(data["rows"])
    picks=manifest["splits"][split][shard::shards]
    blocks=[]
    for i,row in enumerate(picks,1):
        iid=row["id"]
        selected=targets.get(iid)
        if selected is None:
            continue
        pack=torch.load(root/"cache"/split/(str(iid)+".pt"),map_location="cpu",weights_only=True)
        base=make_base(pack,nfine)
        owner={}
        for q,c in zip(base["query"].tolist(),base["label"].tolist()):
            owner.setdefault(q,c)
        query=torch.from_numpy(selected[:,cols["query"]]).long()
        claim=torch.from_numpy(selected[:,cols["class"]]).long()
        old=torch.tensor([owner[q] for q in query.tolist()],dtype=torch.long)
        assert (old!=claim).all()
        image=Image.open(root/"images"/row["dataset"]/row["file_name"]).convert("RGB")
        features=evidence.encode(image,pack,query,torch.stack((old,claim),dim=1))
        keys=selected[:,[cols[k] for k in ("image","query","class","native_score","aligned","strict_fix","old_harmed")]]
        blocks.append(np.column_stack((keys,features)))
        if i%50==0 or i==len(picks):
            print(name,shard,i,len(picks),flush=True)
    table=np.concatenate(blocks)
    assert len(table)==sum(len(targets[r["id"]]) for r in picks if r["id"] in targets)
    fields=("image","query","class","native_score","aligned","strict_fix","old_harmed","v5_p50","v5_p75",
            "extent","spill","inside_agreement","reject","old_prob","new_prob","old_supported")
    assert table.shape[1]==len(fields)
    dest=Path("operational_"+name+"_"+str(shard)+".npz")
    temp=dest.with_suffix(".part")
    with temp.open("wb") as f:
        np.savez_compressed(f,rows=table,columns=np.array(fields))
    temp.replace(dest)
    return dict(actions=len(table),aligned=int(table[:,4].sum()),strict_fix=int(table[:,5].sum()),
                old_harmed=int(table[:,6].sum()),file=str(dest))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--shard",type=int,required=True)
    p.add_argument("--shards",type=int,required=True)
    args=p.parse_args()
    assert json.loads(Path("extent_probe.json").read_text())["passed"]
    assert json.loads(Path("hypothesis_qualification.json").read_text())["passed"]
    assert not json.loads((RELABEL/"eval_deferred.json").read_text())["opened"]
    evidence=Evidence()
    out={}
    for name in ("train","mine","cal"):
        out[name]=process(evidence,name,args.shard,args.shards)
        save_json("operational_status_"+str(args.shard)+".json",out)
        print(out[name],flush=True)
