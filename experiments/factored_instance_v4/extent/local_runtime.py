"""Run the frozen original detector once and reproduce its graph shortlist online."""
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torchvision.ops import box_iou,nms,roi_align

from paired_hypotheses import RelabelEvidence,V1,hypotheses,vocabulary,save_json
from donor_transport_probe import Native,SIZE,actual_donors,transport
from cache import GRAPH_BUDGET,build_graph,clip_crops,xyxy
from counterfactual import candidate_actions,make_base
from relabel import read_factor
from train_hypothesis import SharedHypothesisHead
from joint_gate import JointGate,design,load

HERE=Path(__file__).resolve().parent
FIELDS=("v5_p50","reject","old_prob","new_prob","donor_transport","native_score")
COLUMNS={name:i for i,name in enumerate(FIELDS)}


class Runtime:
    def __init__(self,names,nfine,parents):
        self.native=Native(names)
        self.reader=RelabelEvidence()
        self.reader.vocabulary(names,nfine,parents)
        self.names=names
        self.nfine=nfine
        self.parents=torch.tensor(parents,device="cuda")
        self.alignment=read_factor(HERE.parent/"pooled/best.pt")
        state=torch.load(HERE/"hypothesis_best.pt",map_location="cpu",weights_only=True)
        self.semantic=SharedHypothesisHead(state["width"]).cuda().eval()
        self.semantic.load_state_dict(state["model"])
        self.ranker=JointGate(2,4).eval()
        self.ranker.load_state_dict(torch.load(HERE/"transport_joint_best.pt",map_location="cpu",weights_only=True)["model"])
        with torch.inference_mode():
            t=self.reader.processor(text=[f"a photo of a {s}." for s in names],padding=True,return_tensors="pt").to("cuda")
            self.text=F.normalize(self.reader.clip.get_text_features(**t).float(),dim=-1)

    @torch.inference_mode()
    def native_pack(self,image):
        boxes,scores,field,scale=self.native.forward(image)
        raw=self.native.last_output
        mask=torch.nonzero(((boxes[:,2:]-boxes[:,:2])>0).all(1)).flatten()
        nodes=mask[nms(boxes[mask],scores[mask].amax(-1),.82)[:GRAPH_BUDGET]]
        pack=dict(boxes=boxes.cpu(),logits=scores.cpu(),node_ids=nodes.cpu())
        if len(nodes)<2:
            pack["ids"]=torch.empty(0,dtype=torch.long)
            return pack,field,scale
        subset=boxes[nodes]
        prev=xyxy(raw.intermediate_reference_points[0,-2])[nodes]
        stability=box_iou(subset,prev).diagonal()
        clip=clip_crops(image,subset,self.reader.processor,self.reader.clip)
        query=F.normalize(raw.last_hidden_state[0,nodes].float(),dim=-1)
        i,j,valid,labels,loc,stats,high=build_graph(scores[nodes],subset,stability,clip,clip@self.text.T,self.parents,self.nfine)
        grid=boxes.new_zeros((len(i),25,query.shape[1]))
        if len(i):
            capture=self.native.capture
            h,w=capture["spatial_shapes"][0].tolist()
            visual=capture["vision_encoder_hidden_states"][0,:h*w].reshape(h,w,-1).permute(2,0,1).unsqueeze(0).contiguous()
            pad=.25*(subset[i,2:]-subset[i,:2])
            region=(subset[i]+torch.cat((-pad,pad),1)).clamp(0,1)
            grid=roi_align(visual,[region*scale],5,aligned=True).flatten(2).transpose(1,2)
        pack.update(ids=nodes[i].cpu(),i=i.cpu(),j=j.cpu(),valid=valid.cpu(),labels=labels.cpu(),
                    stats=stats.half().cpu(),high=high[i].cpu(),features=torch.cat((query,clip),dim=1).half().cpu(),
                    grid=grid.half().cpu())
        return pack,field,scale

    @torch.inference_mode()
    def candidate_table(self,pack,field,scale):
        base=make_base(pack,self.nfine)
        actions=[a for a in candidate_actions(pack,base,self.nfine) if a[0]=="replace"]
        if not actions:
            return base,actions,np.empty((0,len(FIELDS))),np.empty(0,dtype=int)
        q=torch.tensor([a[1] for a in actions])
        c=torch.tensor([a[2] for a in actions])
        slot=torch.tensor([a[3] for a in actions])
        old=base["label"][slot]
        assert torch.all(old!=c)
        lookup={int(q):i for i,q in enumerate(pack["ids"])}
        own_grid,own=self.reader.spatial(pack,torch.tensor([lookup[int(i)] for i in q]))
        align=self.alignment.alignment_logits(own_grid.cuda(),own.cuda()).softmax(-1).cpu()[:,1:].sum(-1)
        evidence=hypotheses(self.reader,pack,q,torch.stack((old,c),dim=1))
        pred=self.semantic({k:v.cuda() for k,v in evidence.items()},torch.arange(len(q),device="cuda")).softmax(-1).cpu()
        donors=actual_donors(pack,self.nfine)
        d=torch.tensor([donors[a[1],a[2]] for a in actions])
        keys,inv=torch.unique(torch.cat((q,d)),return_inverse=True)
        patch=roi_align(field,[pack["boxes"][keys].cuda()*scale],SIZE,sampling_ratio=2,aligned=True)
        vec=F.normalize(patch.flatten(2).transpose(1,2),dim=-1)
        ir,id_=inv.chunk(2)
        cover=[]
        for a,b in zip(ir.split(64),id_.split(64)):
            t,_=transport(vec[b]@vec[a].transpose(1,2))
            cover.append(t.cpu())
        table=torch.column_stack((align,pred,torch.cat(cover),base["score"][slot])).numpy()
        utility,_=self.ranker.probabilities(*design(table,COLUMNS))
        order=np.argsort(-utility.numpy(),kind="stable")[:8]
        return base,actions,table,order


@torch.inference_mode()
def smoke():
    manifest=json.loads((V1/"manifest.json").read_text())
    names,nfine,parents=vocabulary(manifest,"cal")
    ref,fields=load("cal","transport")
    engine=Runtime(names,nfine,parents)
    checked=[]
    for row in manifest["splits"]["cal"]:
        if any(a["iscrowd"] for a in row["annotations"]):
            continue
        sample=ref[ref[:,fields["image"]]==row["id"]]
        if len(sample)<8:
            continue
        iid=row["id"]
        expected=torch.load(V1/"cache/cal"/(str(iid)+".pt"),map_location="cpu",weights_only=True)
        image=Image.open(V1/"images"/row["dataset"]/row["file_name"]).convert("RGB")
        pack,field,scale=engine.native_pack(image)
        for name in ("boxes","logits","ids","node_ids","valid","high","labels","grid","features","stats"):
            assert torch.equal(pack[name],expected[name]),(iid,name)
        base,actions,values,top=engine.candidate_table(pack,field,scale)
        idx={(int(a[fields["query"]]),int(a[fields["class"]])):i for i,a in enumerate(sample)}
        assert len(actions)==len(idx)
        match=sample[[idx[a[1],a[2]] for a in actions]][:,[fields[k] for k in FIELDS]]
        diff=np.abs(match-values).max(axis=0)
        assert float(diff.max())<=.002,(iid,diff.tolist())
        u,_=engine.ranker.probabilities(*design(match,COLUMNS))
        expected_top=np.argsort(-u.numpy(),kind="stable")[:8]
        assert np.array_equal(top,expected_top),(iid,top.tolist(),expected_top.tolist())
        check=dict(image=iid,actions=len(actions),native_equal=True,top8_equal=True,max_feature_error=float(diff.max()))
        checked.append(check)
        print(check,flush=True)
        if len(checked)==3:
            break
    assert len(checked)==3
    save_json("runtime_smoke.json",dict(checked=checked,opened_eval=False))


if __name__=="__main__":
    smoke()
