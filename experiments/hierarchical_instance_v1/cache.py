"""Freeze DINO and CLIP, build sparse directed instance graphs without labels."""
import argparse
import json
import math
import os
from pathlib import Path
from time import perf_counter

import torch
from PIL import Image
from torch.nn import functional as F
from torchvision.ops import box_iou, nms, roi_align
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, CLIPModel

GRAPH_BUDGET = 128
K = 4


def xyxy(boxes):
    return torch.cat((boxes[...,:2]-boxes[...,2:]/2,boxes[...,:2]+boxes[...,2:]/2),dim=-1).clamp(0,1)


def vocabulary(manifest,split):
    fine=manifest["base"]+(manifest["novel"] if split=="eval" else [])
    coarse=sorted({p for c,p in manifest["parents"].items() if c in fine})
    names=fine+coarse
    parents=[names.index(manifest["parents"][c]) if c in manifest["parents"] else -1 for c in fine]
    return names,len(fine),parents


def clip_crops(image,boxes,processor,model):
    scale=torch.tensor([image.width,image.height,image.width,image.height],device=boxes.device)
    crops=[]
    for x1,y1,x2,y2 in (boxes*scale).tolist():
        crops.append(image.crop((math.floor(x1),math.floor(y1),math.ceil(x2),math.ceil(y2))))
    chunks=[]
    for start in range(0,len(crops),32):
        p=processor(images=crops[start:start+32],return_tensors="pt",input_data_format="channels_last").pixel_values.to("cuda",dtype=torch.float16)
        chunks.append(model.get_image_features(pixel_values=p).float())
    return F.normalize(torch.cat(chunks),dim=-1)


def build_graph(logits,boxes,stable,clip,fit,parents,nfine):
    scores=logits.sigmoid()
    score,label=scores[:,:nfine].max(-1)
    top_two=scores[:,:nfine].topk(2).values
    column=torch.arange(len(boxes),device=boxes.device)
    clip_leaf=fit[:,:nfine].amax(-1)
    clip_best=fit[column,label]
    high=(score>=.42) & (clip_best>=.17) & (clip_best>=clip_leaf-.045)
    low=(~high | ((top_two[:,0]-top_two[:,1])<.13) | (stable<.75)) & (scores.amax(-1)>=.025)
    edge_sim=clip @ clip.T
    overlaps=box_iou(boxes,boxes)
    common=low[:,None] & (overlaps<.45) & (edge_sim>.45)
    local_leaf=fit[:,label]
    same=common & high[None] & (local_leaf>=.15) & (local_leaf>=clip_leaf[:,None]-.075)
    fine_parent=parents[label]
    pp,pidx=scores[:,nfine:].max(-1)
    pidx=pidx+nfine
    parent_candidate=torch.where(high,fine_parent,pidx)
    parent_source=(high & (fine_parent>=0)) | (~high & (pp>=.48) & (fit[column,pidx]>=.19))
    local_parent=fit[:,parent_candidate.clamp_min(0)]
    family=common & parent_source[None] & (local_parent>=.15)
    strength=edge_sim+.35*score[None]+.40*local_leaf
    strength_parent=edge_sim+.35*scores[column,parent_candidate.clamp_min(0)][None]+.40*local_parent

    def pick(mask,strength):
        _,idx=strength.masked_fill(~mask,-torch.inf).topk(2,dim=-1)
        return idx,mask.gather(1,idx)

    leaf_idx,leaf_ok=pick(same,strength)
    parent_idx,parent_ok=pick(family,strength_parent)
    j=torch.cat((leaf_idx,parent_idx),dim=1)
    valid=torch.cat((leaf_ok,parent_ok),dim=1)
    labels=torch.cat((label[leaf_idx],parent_candidate[parent_idx].clamp_min(0)),dim=1)
    i=torch.nonzero(valid.any(-1)).flatten()
    j,valid,labels=j[i],valid[i],labels[i]
    shape=boxes[:,2:]-boxes[:,:2]
    aspect=shape[:,0]/shape[:,1]
    shape_distance=(aspect[i,None].log()-aspect[j].log()).abs()
    loc=valid & (labels<nfine) & (stable[j]>.75) & (shape_distance<.8)
    stats=torch.stack((edge_sim[i[:,None],j],scores[i[:,None],labels],
                       scores[j,labels],fit[i[:,None],labels],fit[j,labels],
                       stable[j],shape_distance.clamp_max(3)/3,(labels>=nfine).float()),dim=-1)
    return i,j,valid,labels,loc,stats,high


def gt_arrays(row,names,manifest):
    id_to_class={manifest["categories"][c]:i for i,c in enumerate(names) if c in manifest["categories"]}
    objects=[a for a in row["annotations"] if not a["iscrowd"] and a["category_id"] in id_to_class]
    boxes=torch.tensor([a["bbox"] for a in objects],dtype=torch.float32).reshape(-1,4)
    boxes[:,2:]+=boxes[:,:2]
    boxes/=torch.tensor([row["width"],row["height"],row["width"],row["height"]])
    labels=torch.tensor([id_to_class[a["category_id"]] for a in objects],dtype=torch.long)
    return boxes,labels


def measure_training(row,pack,names,nfine,parents,manifest):
    truth,y=gt_arrays(row,names,manifest)
    r=pack["ids"]
    zeros=dict(covered=0,with_edge=0,leaf_opportunity=0,parent_opportunity=0,loc_opportunity=0)
    if not len(y):
        return torch.empty(0,dtype=torch.long),zeros
    # Matching is independent of the graph label; use the same target for both outputs.
    ious=box_iou(pack["boxes"].float(),truth)
    if not len(r):
        zeros["covered"]=int((ious.amax(0)>=.50).sum())
        return torch.empty(0,dtype=torch.long),zeros
    rec=ious[r]
    best_iou,best_gt=rec.max(dim=1)
    all_score=pack["logits"].float().sigmoid()
    base_class=all_score[:,:nfine].argmax(-1)
    selected=[]
    sources=pack["node_ids"][pack["j"]]
    source_iou,source_gt=ious[sources].max(dim=-1)
    for g,gt_class in enumerate(y.tolist()):
        if ious[:,g].max() >= .50:
            zeros["covered"]+=1
        valid=(best_gt==g)&(best_iou>=.35)
        if not valid.any():
            continue
        zeros["with_edge"]+=1
        ranked=rec[:,g].masked_fill(~valid,-1)
        i=int(ranked.argmax())
        # Keep one receiver per GT, including correct-label cases that teach abstention.
        selected.append(i)
        if base_class[r[i]]==gt_class and all_score[r[i],gt_class]>=.42 and best_iou[i]>=.75:
            continue
        donor_y=y[source_gt[i]]
        donor_correct=(base_class[sources[i]]==donor_y)&(source_iou[i]>=.50)&(source_gt[i]!=g)&pack["valid"][i]
        same=donor_correct&(donor_y==gt_class)&(pack["labels"][i]==gt_class)
        parent=parents[gt_class]
        related=donor_correct&(donor_y!=gt_class)&(pack["labels"][i]==parent)&(parent>=nfine)
        related&=torch.tensor([parents[x] for x in donor_y.tolist()])==parent
        zeros["leaf_opportunity"]+=bool(same.any())
        zeros["parent_opportunity"]+=bool(related.any())
        zeros["loc_opportunity"]+=bool((same&pack["loc"][i]&(source_iou[i]>=.70)).any() and best_iou[i]<.80)
    return torch.tensor(selected,dtype=torch.long),zeros


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--splits",nargs="+",default=["train","cal","eval"])
    p.add_argument("--limit",type=int)
    p.add_argument("--shard",type=int,default=0)
    p.add_argument("--shards",type=int,default=1)
    args=p.parse_args()
    manifest=json.loads(Path("manifest.json").read_text())
    model_dir=Path(os.environ["BASE"])/"models"
    dino=AutoModelForZeroShotObjectDetection.from_pretrained(model_dir/"dino",local_files_only=True,disable_custom_kernels=True).cuda().eval()
    clip=CLIPModel.from_pretrained(model_dir/"clip",local_files_only=True,torch_dtype=torch.float16,use_safetensors=True).cuda().eval()
    dp=AutoProcessor.from_pretrained(model_dir/"dino",local_files_only=True)
    cp=AutoProcessor.from_pretrained(model_dir/"clip",local_files_only=True)
    capture={}

    def hook(_module,_args,kwargs):
        capture.update({k:kwargs[k] for k in ("vision_encoder_hidden_states","spatial_shapes","valid_ratios")})

    handle=dino.model.decoder.register_forward_pre_hook(hook,with_kwargs=True)
    start=perf_counter()
    for split in args.splits:
        names,nfine,parents=vocabulary(manifest,split)
        prompt=". ".join(names)+"."
        t=dp.tokenizer(prompt,return_offsets_mapping=True)
        assert len(t.input_ids)<=dino.config.max_text_len,(split,len(t.input_ids))
        spans=[]
        offset=0
        for label in names:
            end=offset+len(label)
            spans.append([k for k,(a,b) in enumerate(t.offset_mapping) if a<b and offset<=a and b<=end])
            assert spans[-1],label
            offset=end+2
        token_positions=torch.tensor([i for span in spans for i in span],device="cuda")
        categories=torch.tensor([k for k,span in enumerate(spans) for _ in span],device="cuda")
        txt=cp(text=[f"a photo of a {name}." for name in names],padding=True,return_tensors="pt").to("cuda")
        text_features=F.normalize(clip.get_text_features(**txt).float(),dim=-1)
        loc_parents=torch.tensor(parents,device="cuda")
        directory=Path("cache")/split
        directory.mkdir(parents=True,exist_ok=True)
        rows=manifest["splits"][split][args.shard::args.shards][:args.limit]
        for index,row in enumerate(rows,1):
            path=directory/(str(row["id"])+".pt")
            if path.is_file():
                continue
            image=Image.open(Path("images")/row["dataset"]/row["file_name"]).convert("RGB")
            inputs=dp(images=image,text=prompt,return_tensors="pt").to("cuda")
            assert inputs.input_ids[0].tolist()==t.input_ids
            output=dino(**inputs)
            out_logits=output.logits[0]
            scores=out_logits.new_full((len(out_logits),len(names)),-torch.inf)
            scores.scatter_reduce_(1,categories.expand(len(scores),-1),out_logits[:,token_positions],reduce="amax")
            boxes=xyxy(output.pred_boxes[0])
            assert torch.isfinite(scores).all() and torch.isfinite(boxes).all()
            size=boxes[:,2:]-boxes[:,:2]
            ok=(size>0).all(-1)
            mask=torch.nonzero(ok).flatten()
            kept=nms(boxes[mask],scores[mask].amax(-1),.82)[:GRAPH_BUDGET]
            nodes=mask[kept]
            pack={"boxes":boxes.cpu(),"logits":scores.cpu(),"node_ids":nodes.cpu()}
            if len(nodes)<2:
                pack.update(ids=torch.empty(0,dtype=torch.long),train_ids=torch.empty(0,dtype=torch.long),screen={})
            else:
                subset=boxes[nodes]
                prev=xyxy(output.intermediate_reference_points[0,-2])[nodes]
                stable=box_iou(subset,prev).diagonal()
                clip_features=clip_crops(image,subset,cp,clip)
                query_features=F.normalize(output.last_hidden_state[0,nodes].float(),dim=-1)
                i,j,valid,labels,loc,stats,high=build_graph(scores[nodes],subset,stable,clip_features,
                                                           clip_features @ text_features.T,loc_parents,nfine)
                n,c=subset.shape[0],query_features.shape[-1]
                grid=boxes.new_zeros((len(i),25,c))
                maps=boxes.new_zeros((len(i),K,25))
                if len(i):
                    h,w=capture["spatial_shapes"][0].tolist()
                    visual=capture["vision_encoder_hidden_states"][0,:h*w].reshape(h,w,c).permute(2,0,1).unsqueeze(0).contiguous()
                    valid_wh=capture["valid_ratios"][0,0]*boxes.new_tensor([w,h])
                    scale=valid_wh.repeat(2)
                    padding=.25*(subset[i,2:]-subset[i,:2])
                    enlarged=(subset[i]+torch.cat((-padding,padding),-1)).clamp(0,1)
                    grid=roi_align(visual,[enlarged*scale],5,aligned=True).flatten(2).transpose(1,2)
                    if loc.any():
                        donor_indices=torch.unique(j[loc],sorted=True)
                        ref=roi_align(visual,[subset[donor_indices]*scale],5,aligned=True).flatten(2).transpose(1,2)
                        r,e=torch.nonzero(loc,as_tuple=True)
                        donor=ref[torch.searchsorted(donor_indices,j[r,e])]
                        maps[r,e]=(F.normalize(grid[r],dim=-1)@F.normalize(donor,dim=-1).transpose(1,2)).amax(-1)
                pack.update(ids=nodes[i].cpu(),i=i.cpu(),j=j.cpu(),valid=valid.cpu(),labels=labels.cpu(),loc=loc.cpu(),
                            stats=stats.half().cpu(),high=high[i].cpu(),features=torch.cat((query_features,clip_features),-1).half().cpu(),
                            grid=grid.half().cpu(),maps=maps.half().cpu())
                if split!="eval":
                    pack["train_ids"],pack["screen"]=measure_training(row,pack,names,nfine,parents,manifest)
            temp=path.with_suffix(".part")
            torch.save(pack,temp)
            temp.replace(path)
            if index%25==0 or index==len(rows):
                print(split,args.shard,index,len(rows),round(perf_counter()-start,1),flush=True)
    handle.remove()


if __name__=="__main__":
    with torch.inference_mode():
        main()
