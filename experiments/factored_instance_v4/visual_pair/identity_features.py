"""Observe native decoder spatial evidence for same-class instance association."""
import argparse
import json

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from dense_donor import ROOT,SHARDS,save_json,vocabulary
from donor_transport_probe import Native
from identity_pairs import PAIR_FIELDS
from operational_cache import sources
from train_factor import by_image

LAYERS = 3
SIDE = 8
FEATURES = ("iou","box_center_distance","abs_log_area_ratio","query_cosine")+tuple(
    f"layer{i}_{k}" for i in range(LAYERS) for k in
    ("support_overlap","support_center_distance","exclusive_ownership","local_mass"))+("low_class_logit","high_class_logit")


class FootprintNative(Native):
    def __init__(self,names):
        super().__init__(names)
        self.offset = {}
        self.footprint = {}
        self.hooks = []
        for i,layer in enumerate(self.model.model.decoder.layers[-LAYERS:]):
            attention = layer.encoder_attn
            self.hooks.append(attention.sampling_offsets.register_forward_hook(self.offset_hook(i)))
            self.hooks.append(attention.register_forward_hook(self.capture_hook(i),with_kwargs=True))

    def offset_hook(self,i):
        def capture(module,args,result):
            self.offset[i] = result
        return capture

    def capture_hook(self,i):
        def capture(module,args,kwargs,result):
            ref = kwargs["reference_points"]
            assert ref.shape[-1]==4 and ref.shape[0]==1
            w = result[1]
            offsets = self.offset.pop(i).reshape(*w.shape,2)
            position = ref[:,:,None,:,None,:2]+offsets/module.n_points*ref[:,:,None,:,None,2:]*.5
            ratios = self.capture["valid_ratios"][0,None,None,:,None,:]
            xy = (position[0]/ratios).float().flatten(1,3)
            mass = w[0].float().flatten(1)/w.shape[2]
            assert torch.isfinite(xy).all() and torch.isfinite(mass).all()
            assert torch.allclose(mass.sum(1),torch.ones_like(mass[:,0]),atol=1e-5)
            self.footprint[i] = xy,mass
        return capture

    def forward(self,image):
        self.footprint.clear()
        result = super().forward(image)
        assert len(self.footprint)==LAYERS and not self.offset
        return result


def splat(position,mass):
    inside = ((position>=0)&(position<=1)).all(-1)
    weight = mass*inside
    at = position.clamp(0,1)*(SIDE-1)
    lo = at.floor().long()
    frac = at-lo
    bins = torch.zeros((len(position),SIDE*SIDE),dtype=mass.dtype,device=mass.device)
    for dx,dy in ((0,0),(0,1),(1,0),(1,1)):
        x = (lo[:,:,0]+dx).clamp_max(SIDE-1)
        y = (lo[:,:,1]+dy).clamp_max(SIDE-1)
        wx = frac[:,:,0] if dx else 1-frac[:,:,0]
        wy = frac[:,:,1] if dy else 1-frac[:,:,1]
        bins.scatter_add_(1,y*SIDE+x,weight*wx*wy)
    total = weight.sum(1)
    center = (weight[:,:,None]*position).sum(1)/total[:,None].clamp_min(1e-8)
    return bins/total[:,None].clamp_min(1e-8),center,total


def exclusive(position,mass,own,other):
    def inside(box):
        return ((position>=box[:,None,:2])&(position<=box[:,None,2:])).all(2)
    in_own,in_other = inside(own),inside(other)
    return (mass*(in_own&~in_other)).sum(1)-(mass*(in_other&~in_own)).sum(1)


@torch.inference_mode()
def pair_features(native,boxes,logits,q,r,c):
    a,b = boxes[q],boxes[r]
    awh,bwh = (a[:,2:]-a[:,:2]).clamp_min(1e-6),(b[:,2:]-b[:,:2]).clamp_min(1e-6)
    lo = torch.minimum(a[:,:2],b[:,:2])
    span = (torch.maximum(a[:,2:],b[:,2:])-lo).clamp_min(1e-6)
    area1,area2 = awh.prod(1),bwh.prod(1)
    inter = (torch.minimum(a[:,2:],b[:,2:])-torch.maximum(a[:,:2],b[:,:2])).clamp_min(0).prod(1)
    iou = inter/(area1+area2-inter)
    gap = (((a[:,:2]+a[:,2:]-b[:,:2]-b[:,2:])*.5)/span).norm(dim=1)
    hidden = native.last_output.last_hidden_state[0].float()
    result = [iou,gap,(area1.log()-area2.log()).abs(),F.cosine_similarity(hidden[q],hidden[r])]
    for i in range(LAYERS):
        points,mass = native.footprint[i]
        x,y = points[q],points[r]
        wx,wy = mass[q],mass[r]
        hx,mx,sx = splat((x-lo[:,None])/span[:,None],wx)
        hy,my,sy = splat((y-lo[:,None])/span[:,None],wy)
        similarity = (hx*hy).clamp_min(0).sqrt().sum(1)
        difference = (mx-my).norm(dim=1)
        ownership = .5*(exclusive(x,wx,a,b)+exclusive(y,wy,b,a))
        result.extend((similarity,difference,ownership,torch.minimum(sx,sy)))
    left,right = logits[q,c],logits[r,c]
    result.extend((torch.minimum(left,right),torch.maximum(left,right)))
    output = torch.stack(result,dim=1)
    assert output.shape==(len(q),len(FEATURES)) and torch.isfinite(output).all()
    return output.cpu().numpy()


@torch.inference_mode()
def cache(split,shard):
    root,tag,_ = sources(split)
    manifest = json.loads((root/"manifest.json").read_text())
    names,nfine,_ = vocabulary(manifest,tag)
    assert nfine==48
    pairs = []
    for flag,prefix in enumerate(("identity_pairs","identity_enriched")):
        with np.load(ROOT/f"{prefix}_{split}.npz") as raw:
            assert tuple(raw["columns"])==PAIR_FIELDS
            data = raw["rows"]
            pairs.append(np.column_stack((data,np.full(len(data),flag))))
    pairs = np.concatenate(pairs)
    pairs = pairs[np.argsort(pairs[:,0],kind="stable")]
    index = by_image(pairs)
    images = [row for row in manifest["splits"][tag] if row["id"] in index][shard::SHARDS]
    native = FootprintNative(names)
    result,features = [],[]
    for j,row in enumerate(images,1):
        iid = row["id"]
        a = index[iid]
        im = Image.open(root/"images"/row["dataset"]/row["file_name"]).convert("RGB")
        boxes,logits,_,_ = native.forward(im)
        pack = torch.load(root/"cache"/tag/f"{iid}.pt",map_location="cpu",weights_only=True)
        assert torch.equal(boxes.cpu(),pack["boxes"]) and torch.equal(logits.cpu(),pack["logits"])
        q,r,c = a[:,[1,4,2]].astype(int).T
        x = pair_features(native,boxes,logits,q,r,c)
        assert np.allclose(x[:,0],a[:,12],atol=2e-6)
        result.append(a)
        features.append(x)
        if j%50==0 or j in (1,len(images)):
            print("identity features",split,shard,j,len(images),flush=True)
    path = ROOT/f"identity_features_{split}_{shard}.npz"
    with path.with_suffix(".part").open("wb") as f:
        np.savez_compressed(f,rows=np.concatenate(result),features=np.concatenate(features),
                            columns=np.array(PAIR_FIELDS+("enriched",)),fields=np.array(FEATURES))
    path.with_suffix(".part").replace(path)


if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--split",choices=("train","mine","cal"),required=True)
    p.add_argument("--shard",type=int,choices=range(SHARDS),required=True)
    a = p.parse_args()
    cache(a.split,a.shard)
