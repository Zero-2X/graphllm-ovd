"""Cache detector and receiver-only CLIP evidence for two category claims."""
import argparse
from collections import Counter
import os
from pathlib import Path

from PIL import Image
import torch
from torch.nn import functional as F
from torchvision.ops import box_iou
from transformers import AutoProcessor, CLIPModel

from objects365_coverage_cache import OUT as COVERAGE, PLAN as COVERAGE_PLAN, native_detector, coverage
from objects365_cache import block_lock, block_path, split_rows
from objects365_data import BLOCK, FOLDER, HERE, INDEX, SHARDS, digest, read
from objects365_edge_cache import OUT as EDGE, PLAN as EDGE_PLAN, split_document
from objects365_instance_gate import ReceiverAttention
from layer_entropy_cache import make_base
from cache import clip_crops, gt_arrays
from counterfactual import crowds, overlaps_for_crowds

OUT = FOLDER/"claim_check"
PLAN = HERE/"OBJECTS365_CLAIM_CHECK.md"


class ReceiverVisual:
    @torch.inference_mode()
    def __init__(self, doc):
        self.native = native_detector(doc)
        self.attention = ReceiverAttention(self)
        path = Path(os.environ["BASE"])/"models"/"clip"
        self.clip = CLIPModel.from_pretrained(path,local_files_only=True,torch_dtype=torch.float16,
                                             use_safetensors=True).cuda().eval().requires_grad_(False)
        self.processor = AutoProcessor.from_pretrained(path,local_files_only=True)
        text = self.processor(text=[f"a photo of a {name}." for name in doc["names"][:365]],
                              padding=True,return_tensors="pt").to("cuda")
        self.text = F.normalize(self.clip.get_text_features(**text).float(),dim=-1)

    def views(self, image, boxes, actions, labels, detector):
        query = actions[:,0]
        unique, inverse = query.unique(return_inverse=True)
        crop = clip_crops(image,boxes[unique].cuda(),self.processor,self.clip)
        focused,_ = self.attention.crops(image,boxes[unique],unique,self)
        all_scores = torch.stack((crop,focused.cuda())) @ self.text.T
        requested = all_scores[:,inverse].gather(-1,labels.cuda()[None].expand(2,-1,-1))
        margin = requested-all_scores[:,inverse].amax(-1,keepdim=True)
        clip = torch.stack((requested,margin),dim=-1).transpose(0,1).cpu()
        native = detector.reshape(len(actions),4,2).transpose(1,2)
        return torch.cat((native[:,None].expand(-1,2,-1,-1),clip),dim=-1).half()


def claim_labels(frame, doc, boxes, labels):
    truth,category = gt_arrays(frame,doc["names"],doc)
    overlap = box_iou(boxes,truth)
    aligned = (overlap>=.5).any(1)
    class_match = category[None,None,:]==labels[:,:,None]
    class_overlap = (overlap[:,None]*class_match).amax(-1) if overlap.shape[1] else boxes.new_zeros(labels.shape)
    crowded,crowd_category = crowds(frame,doc,doc["names"])
    crowd_cover = torch.from_numpy(overlaps_for_crowds(boxes,crowded))
    crowd_labels = torch.from_numpy(crowd_category)
    ignore = ((crowd_cover[:,None]>=.5) & (crowd_labels[None,None,:]==labels[:,:,None])).any(-1)
    target = torch.full_like(labels,-1)
    target[(class_overlap<=.3) & aligned[:,None] & ~ignore] = 0
    target[(class_overlap>=.5) & ~ignore] = 1
    return target,class_overlap


@torch.inference_mode()
def extract(doc, phase, shard):
    OUT.mkdir(parents=True,exist_ok=True)
    visual = ReceiverVisual(doc)
    splits = ("train","rare_train","monitor") if phase=="fit" else ("policy","eval")
    for split in splits:
        source,original,index = split_document(doc,split)
        frames = {row["id"]:row for row in source["rows"]}
        rows = split_rows(source,original,shard)
        for offset in range(0,len(rows),BLOCK):
            part = offset//BLOCK
            name = f"{split}_{shard}_{part:03d}.pt"
            lock = block_lock(index,rows[offset:offset+BLOCK])
            lock.update(plan=digest(EDGE_PLAN),encoder=digest(HERE/"objects365_instance_gate.py"))
            context_key = dict(edge=lock,method=digest(COVERAGE_PLAN))
            key = dict(context=context_key,method=digest(PLAN))
            target = OUT/name
            if target.exists():
                assert torch.load(target,map_location="cpu",weights_only=True)["key"]==key
                continue
            cached = torch.load(EDGE/name,map_location="cpu",weights_only=True)
            context = torch.load(COVERAGE/name,map_location="cpu",weights_only=True)
            assert cached["lock"]==lock and context["key"]==context_key
            baseline = {}
            if split=="eval":
                baseline = {row["image"]:row for row in torch.load(block_path(split,shard,part),
                            map_location="cpu",weights_only=True)["records"]}
            features,summary = {},Counter()
            for record in cached["records"]:
                image_id = record["image"]
                if image_id not in context["records"]:
                    continue
                frame = frames[image_id]
                actions = record["actions"]
                query,proposed,slot = actions.T
                with Image.open(frame["path"]) as source_image:
                    image = source_image.convert("RGB")
                box,logits,_,_ = visual.native.forward(image)
                pack = dict(boxes=box.float().cpu(),logits=logits.float().cpu())
                base = make_base(pack,365)
                if split=="eval":
                    expected = baseline[image_id]
                    if not torch.equal(pack["boxes"],expected["boxes"]) or not all(
                            torch.equal(base[field],expected["base"][field]) for field in base):
                        summary["skipped"] += 1
                        continue
                if int(slot.max())>=len(base["query"]) or not torch.equal(base["query"][slot],query) or (base["label"][slot]==proposed).any():
                    summary["skipped"] += 1
                    continue
                detector = coverage(pack,base,actions)
                if not torch.allclose(detector,context["records"][image_id],atol=.0001,rtol=.0001):
                    summary["skipped"] += 1
                    continue
                labels = torch.column_stack((base["label"][slot],proposed))
                inputs = visual.views(image,pack["boxes"],actions,labels,detector)
                y,ious = claim_labels(frame,doc,pack["boxes"][query],labels)
                assert inputs.shape==(len(actions),2,2,6) and inputs.isfinite().all()
                features[image_id] = dict(actions=actions,labels=labels,x=inputs,y=y,ious=ious)
                summary["images"] += 1
                summary["actions"] += len(actions)
            torch.save(dict(key=key,features=features,counts=dict(summary)),target.with_suffix(".part"))
            target.with_suffix(".part").replace(target)
            print(split,shard,part,dict(summary),flush=True)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase",choices=("fit","test"))
    parser.add_argument("--shard",type=int,required=True,choices=range(SHARDS))
    args = parser.parse_args()
    extract(read(INDEX),args.phase,args.shard)
