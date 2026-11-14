"""One frozen native forward, exact historical features, and predicted-box local9."""
import torch
from torch.nn import functional as F
from torchvision.ops import box_iou

from graphllm_local_message_unique_v2.bridge import Vision as FrozenVision
from graphllm_local_message_unique_v2.core import edge_geometry


def grid_xy(size, device=None):
    side=(torch.arange(size,device=device,dtype=torch.float32)+.5)/size
    y,x=torch.meshgrid(side,side,indexing="ij")
    return torch.stack((x,y),-1).flatten(0,1)


def sample_local(feature,mask,ratios,boxes,size,min_weight):
    xy=grid_xy(size,feature.device)
    points=boxes[:,None,:2]+xy*(boxes[:,None,2:]-boxes[:,None,:2])
    grid=(2*points*ratios-1)[None]
    valid=mask.to(torch.float32)[None,None]
    norm=F.grid_sample(valid,grid,mode="bilinear",align_corners=False,padding_mode="zeros")[0,0]
    value=F.grid_sample(feature.float()[None]*valid,grid,mode="bilinear",
                        align_corners=False,padding_mode="zeros")[0].permute(1,2,0)
    ok=((norm>min_weight).all(1)&torch.isfinite(value).all(dim=(1,2))&
        torch.isfinite(boxes).all(1)&(boxes[:,2:]>boxes[:,:2]).all(1))
    return (value/norm.clamp_min(min_weight)[...,None]).cpu(),ok.cpu()


class LocalReader:
    def __init__(self,native,cfg):
        self.cfg=cfg
        core=native.model.model
        assert not core.config.use_timm_backbone
        backbone=core.backbone.conv_encoder.model
        bc=backbone.config
        assert bc.model_type=="swin" and bc.patch_size==4
        stages=tuple(bc.out_features)
        strides={name:bc.patch_size*2**(i-1) for i,name in enumerate(backbone.stage_names) if i>0}
        stride=cfg["local"]["nominal_stride"]
        self.level=[strides[n] for n in stages].index(stride)
        assert list(bc.out_indices)==[backbone.stage_names.index(n) for n in stages]
        self.identity=dict(stride=stride,level=self.level,stage=stages[self.level],
                           feature_levels=list(stages),strides=[strides[n] for n in stages],
                           tensor="GroundingDinoDecoder.vision_encoder_hidden_states",
                           layout="level_start_index/spatial_shapes",mask="vision_encoder_attention_mask: true=valid",
                           coordinate="native normalized unpadded xyxy, valid_ratios applied once",
                           mode="bilinear/zeros/align_corners_false/valid_mask_renorm")
        self.handle=core.decoder.register_forward_pre_hook(self.capture,with_kwargs=True)
        self.current=None

    def capture(self,_module,_args,kw):
        k=self.level
        starts,shapes=kw["level_start_index"],kw["spatial_shapes"]
        memory=kw["vision_encoder_hidden_states"]
        assert len(memory)==1
        h,w=map(int,shapes[k].tolist())
        start=int(starts[k])
        assert start==int(shapes[:k].prod(1).sum())
        f=memory[0,start:start+h*w].T.reshape(-1,h,w)
        mask=kw["vision_encoder_attention_mask"][0,start:start+h*w].reshape(h,w).bool()
        ratios=kw["valid_ratios"][0,k].float()
        exact=torch.stack((mask[0].float().mean(),mask[:,0].float().mean()))
        torch.testing.assert_close(ratios,exact,rtol=0,atol=1e-6)
        assert int(mask.sum())==round(float(ratios[0])*w)*round(float(ratios[1])*h)
        assert f.shape[0]==self.cfg["dimensions"]["local"]
        self.current=f,mask,ratios

    def __call__(self,boxes):
        assert self.current is not None
        f,mask,ratios=self.current
        self.current=None
        if not len(boxes):
            return torch.empty(0,self.cfg["dimensions"]["grid"]**2,f.shape[0]),torch.empty(0,dtype=torch.bool)
        return sample_local(f,mask,ratios,boxes.to(f.device),self.cfg["dimensions"]["grid"],self.cfg["local"]["min_weight"])


def canonical_key(raw,base,image,cat):
    q,c,s=map(int,raw)
    return image,q,s,cat[int(base["label"][s])],cat[c]


def attach_local(scan,local,ok,frame,cat):
    pack,base=scan["pack"],scan["base"]
    index={int(q):i for i,q in enumerate(pack["node_ids"])}
    actions=[]
    rejected=[]
    for a in scan["actions"]:
        key=a["key"]
        ids=[index[q] for q in [key[1]]+a["donor_query_ids"]]
        key=(*key[:3],cat[key[3]],cat[key[4]])
        if not a["donor_query_ids"]:
            rejected.append(dict(key=key,reject_reason="no_raw_donor"))
        elif not all(bool(ok[i]) for i in ids):
            rejected.append(dict(key=key,reject_reason="invalid_local_feature"))
        elif not torch.isfinite(local[ids]).all():
            rejected.append(dict(key=key,reject_reason="invalid_local_feature"))
        else:
            actions.append(dict(key=key,receiver=ids[0],donors=ids[1:],edge=a["edges"],
                                native=a["native5"],native16=a["native16"],
                                donor_query_ids=a["donor_query_ids"],donor_predicted_classes=[key[4]]*len(ids[1:])))
    rejected.extend(dict(key=(*r["key"][:3],cat[r["key"][3]],cat[r["key"][4]]),reject_reason=r["reject_reason"])
                    for r in scan["rejected"])
    raw=[canonical_key(r,base,frame["id"],cat) for r in scan["raw"]]
    assert len(raw)==len(actions)+len(rejected)
    nodes=pack.get("features",torch.empty(len(local),768))
    return dict(image=frame["id"],base=base,raw=raw,actions=actions,rejected=rejected,
                local=local.half(),nodes=nodes,queries=pack["node_ids"])


def concrete_pool(row,pack,cfg):
    if not row["actions"]:
        return None
    b=pack["boxes"][row["queries"]]
    leaf_count=cfg["classes"]
    scores=pack["logits"][row["queries"],:leaf_count].sigmoid()
    high_score,labels=scores.max(1)
    fit=pack["fit"][:,:leaf_count]
    selected=fit.gather(1,labels[:,None]).flatten()
    t=cfg["candidate"]
    best=fit.amax(1)
    high=(high_score>=t["donor_min"])&(selected>=t["donor_clip"])&(selected>=best-t["donor_clip_gap"])
    sim=pack["clip_full"]
    allowed=(high[None]&(sim>t["edge_cos_min"])&(box_iou(b,b)<t["edge_iou_max"]))
    allowed.fill_diagonal_(False)
    return b,high_score,labels,fit,best,sim,allowed


def raw_extension(row,action,pack,pool,cfg,leaf):
    """Enumerate only the original concrete pre-topk pool for a train action."""
    b,high_score,labels,fit,best,sim,mask=pool
    queries=row["queries"]
    ri=action["receiver"]
    br=b[ri]
    c=leaf[action["key"][4]]
    t=cfg["candidate"]
    if fit[ri,c]<t["receiver_clip"] or fit[ri,c]<best[ri]-t["receiver_clip_gap"]:
        return []
    allowed=mask[ri]&(labels==c)
    sim=sim[ri]
    for j in action["donors"]:
        allowed[j]=False
    strength=sim+.35*high_score+.40*fit[ri,c]
    out=[]
    for j in sorted(allowed.nonzero().flatten().tolist(),key=lambda j:(-float(strength[j]),int(queries[j]))):
        if not torch.isfinite(row["local"][j]).all() or not row["local_ok"][j]:
            continue
        aspect=(br[2:]-br[:2]).log().diff()-(b[j,2:]-b[j,:2]).log().diff()
        basic=torch.stack((sim[j],high_score[j],fit[j,c],pack["stability"][j],aspect.abs().clamp_max(3)[0]/3,
                           br.new_zeros(())))
        edge=torch.cat((basic,edge_geometry(br,b[j])))
        if torch.isfinite(edge).all() and torch.isfinite(row["nodes"][j]).all():
            out.append((j,int(queries[j]),edge))
    return out


class NativeBridge:
    def __init__(self,doc,cfg):
        self.vision=FrozenVision(doc)
        self.local=LocalReader(self.vision.graph.native,cfg)
        self.categories=[doc["categories"][s] for s in doc["base"]]
        self.risk=self.vision.risk.requires_grad_(False).eval()
        assert len(self.categories)==cfg["classes"]

    @torch.inference_mode()
    def scan(self,frame,training_pool=False):
        assert set(frame)=={"id","source_id","file_name","path","width","height","sha"}
        projected=[]
        hook=None
        if training_pool:
            hook=self.vision.visual.clip.visual_projection.register_forward_hook(
                lambda _m,_args,y:projected.append(y))
        try:
            scan=self.vision.scan(frame)
        finally:
            if hook is not None:
                hook.remove()
        pack=scan["pack"]
        if training_pool and len(pack["ids"]):
            n=len(pack["node_ids"])
            clip=F.normalize(torch.cat(projected,0)[:n].float())
            assert torch.equal(clip.half().cpu(),pack["features"][:,256:])
            pack["clip_full"]=(clip@clip.T).cpu()
        local,ok=self.local(pack["boxes"][pack["node_ids"]])
        row=attach_local(scan,local,ok,frame,self.categories)
        row["local_ok"]=ok
        row["timing"]=dict(native=scan["native_time"],visual=scan["feature_time"])
        return row,scan
