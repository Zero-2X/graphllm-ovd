"""Build verified category replacement supervision from OmniLabel only."""
import argparse
from collections import Counter, defaultdict
import hashlib
import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from layer_entropy_cache import HERE, NATURAL, ImageReplay, RelabelEvidence, digest, encode, expanded, make_base, natural_setup, read, vocabulary
from local_runtime import Runtime
from donor_transport_probe import Native
from semantic_action_pool import eligible_actions
from route_screen import save_json


FOLDER = HERE / "omni_layer"
PLAN = HERE / "OMNI_LAYER_ENTROPY_PLAN_ZH.md"
INDEX = FOLDER / "manifest.json"
SHARDS = 8
DATA = Path(os.environ["BASE"]) / "data/omnilabel"


def normalize(s):
    return " ".join(s.casefold().strip().split())


def prepare():
    FOLDER.mkdir(exist_ok=True)
    coco, _, _ = natural_setup()
    names, nfine, parents = vocabulary(coco, "cal_policy")
    lookup = {normalize(n):coco["categories"][n] for n in names[:nfine]}
    coco_rows = coco["splits"]["cal_policy"]
    protected = {digest(NATURAL / "images" / r["dataset"] / r["file_name"]) for r in coco_rows}
    assert len(protected) == len(coco_rows)
    seen_path, seen_content, rows, counts, inputs = set(), set(), [], Counter(), {}
    for version in ("v0.1.4", "v0.1.3"):
        path = DATA / "annotations" / f"dataset_all_val_{version}.json"
        inputs[path.name] = digest(path)
        doc = read(path)
        images = DATA / f"validation_{version}" / "images"
        queries, classes = defaultdict(set), {}
        for desc in doc["descriptions"]:
            if desc["anno_info"]["type"] != "object_category":
                continue
            category = lookup.get(normalize(desc["text"]))
            if category is not None:
                classes[desc["id"]] = category
                for iid in desc["image_ids"]:
                    queries[iid].add(category)
        positives = defaultdict(dict)
        for ann in doc["annotations"]:
            for cat in {classes[d] for d in ann["description_ids"] if d in classes} & queries[ann["image_id"]]:
                positives[ann["image_id"]][ann["id"],cat] = dict(id=ann["id"], bbox=ann["bbox"], category_id=cat, iscrowd=ann["iscrowd"])
        for im in doc["images"]:
            rel = im["file_name"]
            source = rel.split("/")[0]
            if source not in ("object365", "openimagesv5"):
                counts["coco_origin_held_out"] += 1
                continue
            if rel in seen_path:
                counts["duplicate_paths"] += 1
                continue
            file = images / rel
            if not file.is_file():
                counts["missing_images"] += 1
                continue
            # An ineligible preferred version does not prevent use of a richer older annotation.
            known = queries[im["id"]]
            ann = list(positives[im["id"]].values())
            if len(known) < 2 or not any(not a["iscrowd"] for a in ann):
                counts["insufficient_queries"] += 1
                continue
            seen_path.add(rel)
            sha = digest(file)
            if sha in protected or sha in seen_content:
                counts["duplicate_bytes_or_coco"] += 1
                continue
            seen_content.add(sha)
            bucket = int(sha[:8],16) % 100
            split = "train" if bucket < 70 else "monitor" if bucket < 85 else "policy"
            rows.append(dict(id=len(rows), official_id=im["id"], path=str(file), file_name=rel, version=version, sha=sha,
                             split=split, source=source, known=sorted(known), annotations=ann))
            counts[f"{split}_{source}"] += 1
    result = dict(plan=digest(PLAN), annotations=inputs, names=names, base=coco["base"], categories=coco["categories"], parents=parents,
                  counts=dict(counts), rows=rows, coco_validation_hashes=len(protected), eval_opened=False)
    save_json(INDEX, result)
    print("OmniLabel index", len(rows), dict(counts), flush=True)


class GraphPack(Runtime):
    def __init__(self, names, nfine, parents):
        self.native = Native(names)
        self.reader = RelabelEvidence()
        self.reader.vocabulary(names, nfine, parents)
        self.nfine = nfine
        self.parents = torch.tensor(parents, device="cuda")
        text = self.reader.processor(text=[f"a photo of a {name}." for name in names], padding=True, return_tensors="pt").to("cuda")
        self.text = F.normalize(self.reader.clip.get_text_features(**text).float(), dim=-1)


def adjudicate(frame, pack, base, doc, actions):
    replay = ImageReplay(frame, pack, base, doc, doc["names"])
    known = set(frame["known"])
    by_id = doc["categories"]
    names = doc["names"]
    result = []
    for q, proposed, slot in actions.tolist():
        old = int(base["label"][slot])
        original_known, proposed_known = by_id[names[old]] in known, by_id[names[proposed]] in known
        support = [float(replay.iou[q,replay.y == c].max(initial=0)) for c in (old,proposed)]
        crowd = ((replay.ioa[q] >= .50) & np.isin(replay.cy,[old,proposed])).any()
        y, wrong, lost, new, fp = -1, 0, [0]*6, [0]*6, [0]*6
        if original_known and proposed_known and not crowd:
            changes = replay.delta(("replace",q,proposed,slot), float(replay.bs[slot]))
            new, lost, fp = [len(v) for v in changes[0]], [len(v) for v in changes[1]], changes[2]
            wrong = int(support[0] >= .50 and support[1] < .50)
            if wrong or any(lost) or (replay.iou[q].max(initial=0) >= .50 and support[1] <= .30):
                y = 0
            elif support[0] < .50 and support[1] >= .50:
                y = 1
        result.append(dict(target=y, correct=int(y==1), wrong=wrong, lost=lost, new=new, fp=fp,
                           unsupported=int(not (original_known and proposed_known) or max(support)<.50)))
    return result


def block(shard, part, rows, doc, engine):
    path = FOLDER / f"data_{shard}_{part:03d}.pt"
    key = dict(index=digest(INDEX), images=[r["id"] for r in rows])
    if path.exists():
        cache = torch.load(path,map_location="cpu",weights_only=True)
        assert cache["lock"] == key
        return
    counts, result = Counter(), []
    for frame in rows:
        source = Path(frame["path"])
        assert digest(source) == frame["sha"]
        with Image.open(source) as raw:
            image = raw.convert("RGB")
        frame = dict(frame,width=image.width,height=image.height)
        pack, _, _ = engine.native_pack(image)
        base = make_base(pack, engine.nfine)
        keys = eligible_actions(pack, base, engine.nfine)
        if not keys:
            counts["no_candidate"] += 1
            continue
        actions = torch.tensor(keys)[:,[0,1,3]]
        if frame["split"] == "policy":
            q,c,_ = actions.T
            top = np.lexsort((c.numpy(),q.numpy(),-pack["logits"][q,c].numpy()))[0]
            actions = actions[top:top+1]
        audit = adjudicate(frame,pack,base,doc,actions)
        for item in audit:
            counts[f'{frame["split"]}_{item["target"]}'] += 1
        keep = [i for i,r in enumerate(audit) if r["target"] >= 0 or frame["split"] == "policy"]
        if not keep:
            continue
        actions = actions[keep]
        views = expanded(engine.reader,source,pack,actions) if frame["split"] == "policy" else None
        result.append(dict(image=frame["id"], split=frame["split"], actions=actions, x=encode(engine.reader,pack,base,actions,views),
                           outcomes=[audit[i] for i in keep]))
    torch.save(dict(lock=key,records=result,counts=dict(counts),eval_opened=False),path.with_suffix(".part"))
    path.with_suffix(".part").replace(path)
    print(shard,part,dict(counts),flush=True)


@torch.inference_mode()
def run(shard, smoke=False):
    doc = read(INDEX)
    assert doc["plan"] == digest(PLAN)
    engine = GraphPack(doc["names"], len(doc["base"]), doc["parents"])
    rows = doc["rows"][shard::SHARDS]
    if smoke:
        # Separate prefix and filename from the full run.
        block(shard,998,rows[:8],doc,engine)
        return
    for start in range(0,len(rows),32):
        block(shard,start//32,rows[start:start+32],doc,engine)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=("prepare","run","smoke"))
    parser.add_argument("--shard",type=int,default=0)
    args=parser.parse_args()
    prepare() if args.mode == "prepare" else run(args.shard,args.mode == "smoke")
