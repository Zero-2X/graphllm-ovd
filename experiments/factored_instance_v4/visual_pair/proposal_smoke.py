"""Check source-conditioned reseeding of the frozen detector, without scoring a test set."""
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torchvision.ops import box_iou,nms

HERE=Path(__file__).resolve().parent
EXTENT=HERE.parent/"extent"
sys.path.insert(0,str(EXTENT))
from donor_transport_probe import Native,actual_donors
from paired_hypotheses import V1,save_json,vocabulary
from cache import xyxy


class ProposalEngine:
    def __init__(self,names,nfine):
        self.native=Native(names)
        self.model=self.native.model
        self.nfine=nfine
        self.decoder_args={}
        self.encoder_query=None
        def save_decoder(_module,_args,kwargs):
            self.decoder_args=dict(kwargs)
        def save_encoder(_module,args,kwargs):
            self.encoder_query=args[0] if args else kwargs["vision_hidden_state"]
        self.decoder_handle=self.model.model.decoder.register_forward_pre_hook(save_decoder,with_kwargs=True)
        self.encoder_handle=self.model.model.encoder_output_class_embed.register_forward_pre_hook(save_encoder,with_kwargs=True)

    def read(self,decoded,args):
        h=decoded.intermediate_hidden_states[:,-1]
        reference=torch.special.logit(decoded.intermediate_reference_points[:,-2],eps=1e-5)
        boxes=(self.model.bbox_embed[-1](h)+reference).sigmoid()
        logits=self.model.class_embed[-1](h,args["text_encoder_hidden_states"],~args["text_encoder_attention_mask"])
        return boxes,logits

    @torch.inference_mode()
    def encode(self,image,pack):
        boxes,scores,_,_=self.native.forward(image)
        raw=self.native.last_output
        assert torch.equal(boxes.cpu(),pack["boxes"]) and torch.equal(scores.cpu(),pack["logits"])
        args=dict(self.decoder_args)
        replica=self.model.model.decoder(**args)
        b,c=self.read(replica,args)
        finite=torch.isfinite(raw.logits)
        assert torch.equal(torch.isfinite(c),finite)
        error=dict(bbox=float((b-raw.pred_boxes).abs().max()),logits=float((c[finite]-raw.logits[finite]).abs().max()))
        assert error["bbox"]<=1e-6 and error["logits"]<=1e-5,error
        assert self.encoder_query is not None and self.model.config.two_stage
        qweight=self.model.model.query_position_embeddings.weight[None]
        assert torch.equal(qweight,args["inputs_embeds"])
        return raw,args,error

    @torch.inference_mode()
    def pool(self,raw,d,c):
        text=raw.encoder_last_hidden_state_text[0]
        pos=self.native.tok[self.native.cat==c]
        t=F.normalize(text[pos].float().mean(0),dim=0)
        teacher=F.normalize(raw.intermediate_hidden_states[0,-1,d].float(),dim=0)
        content=F.normalize(self.encoder_query[0].float(),dim=-1)
        score=torch.minimum(content@t,content@teacher)
        coord=raw.enc_outputs_coord_logits[0]
        a=xyxy(coord.sigmoid())
        valid=torch.isfinite(coord).all(1)&((a[:,2:]-a[:,:2])>0).all(1)
        valid&=box_iou(a,xyxy(raw.pred_boxes[0,d:d+1])).flatten()<.30
        return coord,a,valid,score

    @torch.inference_mode()
    def propose(self,raw,args,d,c):
        coord,a,valid,score=self.pool(raw,d,c)
        k=torch.nonzero(valid).flatten()
        select=k[nms(a[k],score[k],.50)[:16]]
        assert len(select)==16
        return self.decode_seeds(args,coord[select])

    def decode_seeds(self,args,seeds):
        new=dict(args)
        n=len(seeds)
        new["reference_points"]=seeds.sigmoid()[None]
        new["inputs_embeds"]=args["inputs_embeds"][:,:n]
        decoder=self.model.model.decoder
        old_count=decoder.config.num_queries
        try:
            # The pinned implementation sizes its text mask from this field.
            decoder.config.num_queries=n
            out=decoder(**new)
        finally:
            decoder.config.num_queries=old_count
        boxes,logits=self.read(out,new)
        z=logits[0]
        p=z.new_full((len(z),self.native.n),-torch.inf)
        p.scatter_reduce_(1,self.native.cat.expand(len(z),-1),z[:,self.native.tok],reduce="amax")
        assert torch.isfinite(p).all() and torch.isfinite(boxes).all()
        return xyxy(boxes[0]),p[:,:self.nfine].sigmoid()


@torch.inference_mode()
def main():
    manifest=json.loads((V1/"manifest.json").read_text())
    names,nfine,_=vocabulary(manifest,"cal")
    selected={223648,217886,10784}
    case=json.loads((EXTENT/"local_case_trace.json").read_text())["cases"]
    trace={}
    for x in case:
        if x["image"] in selected:
            trace.setdefault(x["image"],x)
    engine=ProposalEngine(names,nfine)
    checked=[]
    for row in manifest["splits"]["cal"]:
        iid=row["id"]
        if iid not in trace:
            continue
        pack=torch.load(V1/"cache/cal"/(str(iid)+".pt"),map_location="cpu",weights_only=True)
        q=trace[iid]["query"]
        c=names.index(trace[iid]["proposed"])
        d=actual_donors(pack,nfine)[q,c]
        im=Image.open(V1/"images"/row["dataset"]/row["file_name"]).convert("RGB")
        raw,args,zero=engine.encode(im,pack)
        boxes,prob=engine.propose(raw,args,d,c)
        io=box_iou(boxes,pack["boxes"][q:q+1].cuda()).flatten()
        p,chosen=prob.max(1)
        matched=(io>=.50)&(chosen==c)&(p>=.05)
        item=dict(image=iid,receiver=q,donor=d,proposed=names[c],zero=zero,new_queries=len(boxes),
                  best_receiver_overlap=float(io.max()),matching_proposed=int(matched.sum()))
        checked.append(item)
        print(item,flush=True)
    assert len(checked)==3
    save_json("proposal_smoke.json",dict(cases=checked,scope="native interface smoke on opened development only"))


if __name__=="__main__":
    main()
