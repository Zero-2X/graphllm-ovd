"""Native detector evidence for whether a proposed edit removes or adds instance coverage."""
import argparse
from collections import Counter

import torch
from PIL import Image
from torchvision.ops import box_iou

from objects365_cache import block_lock, block_path, split_rows, text_capacity
from objects365_data import BLOCK, FOLDER, HERE, INDEX, SHARDS, digest, read, save_json
from objects365_edge_cache import OUT as EDGE, PLAN as EDGE_PLAN, split_document
from objects365_edge_mlp import Head, edge_records, rank
from objects365_fit import collect
from layer_entropy_cache import make_base
from donor_transport_probe import Native

OUT = FOLDER/"coverage"
PLAN = HERE/"OBJECTS365_COVERAGE.md"
FIELDS = ("own_old","own_new","native_local_old","native_local_new",
          "output_local_old","output_local_new","output_iou_old","output_iou_new")


def coverage(pack, base, actions):
    query, proposed, slot = actions.T
    assert torch.equal(base["query"][slot],query)
    old = base["label"][slot]
    assert (old!=proposed).all()
    classes = torch.stack((old,proposed),dim=1)
    scores = pack["logits"].float().sigmoid()
    own = scores[query[:,None],classes]
    overlap = box_iou(pack["boxes"][query],pack["boxes"])
    overlap.scatter_(1,query[:,None],0)
    local = (scores.T[classes]*overlap[:,None]).amax(-1)
    final_overlap = box_iou(pack["boxes"][query],base["box"])
    final_overlap.scatter_(1,slot[:,None],0)
    same_class = classes[:,:,None]==base["label"][None,None]
    final_overlap = final_overlap[:,None]*same_class
    final_score = (final_overlap*base["score"]).amax(-1)
    return torch.cat((own,local,final_score,final_overlap.amax(-1)),dim=1)


def native_detector(doc):
    detector = Native(doc["names"])
    text_capacity(detector,len(detector.ids))
    return detector


@torch.inference_mode()
def native_evidence(detector, frame):
    with Image.open(frame["path"]) as image:
        boxes, logits, _, _ = detector.forward(image.convert("RGB"))
    pack = dict(boxes=boxes.cpu().float(),logits=logits.cpu().float())
    return pack,make_base(pack,365)


@torch.inference_mode()
def audit(doc):
    OUT.mkdir(parents=True,exist_ok=True)
    receipt = read(EDGE/"qualification.json")
    assert receipt["passed"] and receipt["checkpoint"]==digest(EDGE/receipt["model"])
    model = Head().cuda().eval()
    model.load_state_dict(torch.load(EDGE/receipt["model"],map_location="cpu",weights_only=True)["model"])
    records,_ = collect(doc,"policy")
    features,_ = edge_records(doc,"policy")
    rows,_,_ = rank(records,features,model)
    verified = sorted((r for r in rows if r["strict"]>=0),key=lambda row:-row["score"])
    picks = [(row,"harm") for row in verified if row["strict"]==0][:12]
    picks += [(row,"gain") for row in verified if row["strict"]==3]
    detector = native_detector(doc)
    out = []
    for row,kind in picks:
        pack,base = native_evidence(detector,doc["rows"][row["image"]])
        actions = torch.tensor([[row["query"],row["proposed"],row["slot"]]])
        values = coverage(pack,base,actions)[0].tolist()
        original = int(base["label"][row["slot"]])
        out.append(dict(image=row["image"],kind=kind,score=row["score"],
                        original=doc["names"][original],proposed_name=doc["names"][row["proposed"]],
                        evidence=dict(zip(FIELDS,values))))
    save_json(OUT/"audit.json",out)
    print(out,flush=True)


@torch.inference_mode()
def extract(doc, phase, shard):
    OUT.mkdir(parents=True,exist_ok=True)
    detector = native_detector(doc)
    splits = ("train","rare_train","monitor") if phase=="fit" else ("policy","eval")
    for split in splits:
        source_doc, original, index = split_document(doc,split)
        frames = {row["id"]:row for row in source_doc["rows"]}
        rows = split_rows(source_doc,original,shard)
        for offset in range(0,len(rows),BLOCK):
            lock = block_lock(index,rows[offset:offset+BLOCK])
            lock.update(plan=digest(EDGE_PLAN),encoder=digest(HERE/"objects365_instance_gate.py"))
            part = offset//BLOCK
            target = OUT/f"{split}_{shard}_{part:03d}.pt"
            key = dict(edge=lock,method=digest(PLAN))
            if target.exists():
                assert torch.load(target,map_location="cpu",weights_only=True)["key"]==key
                continue
            source = torch.load(EDGE/f"{split}_{shard}_{part:03d}.pt",map_location="cpu",weights_only=True)
            assert source["lock"]==lock
            baseline = {}
            if split=="eval":
                baseline = {row["image"]:row for row in torch.load(block_path(split,shard,part),
                            map_location="cpu",weights_only=True)["records"]}
            records = {}
            skipped = []
            for row in source["records"]:
                frame = frames[row["image"]]
                pack,base = native_evidence(detector,frame)
                if split=="eval":
                    cached = baseline[row["image"]]
                    if not torch.equal(pack["boxes"],cached["boxes"]) or not all(
                            torch.equal(base[field],cached["base"][field]) for field in base):
                        skipped.append(row["image"])
                        continue
                query,proposed,slot = row["actions"].T
                if int(slot.max())>=len(base["query"]) or not torch.equal(base["query"][slot],query) or (base["label"][slot]==proposed).any():
                    skipped.append(row["image"])
                    continue
                value = coverage(pack,base,row["actions"])
                assert value.shape==(len(row["actions"]),len(FIELDS)) and value.isfinite().all()
                records[row["image"]] = value
            torch.save(dict(key=key,records=records,skipped=skipped),target.with_suffix(".part"))
            target.with_suffix(".part").replace(target)
            print(split,shard,part,len(records),skipped,flush=True)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase",choices=("audit","fit","test"))
    parser.add_argument("--shard",type=int,default=0,choices=range(SHARDS))
    args = parser.parse_args()
    doc = read(INDEX)
    if args.phase=="audit":
        audit(doc)
    else:
        extract(doc,args.phase,args.shard)
