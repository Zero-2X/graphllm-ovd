"""Re-extract frozen, edge-preserving Qwen features for the Objects365 MLP."""
import argparse
from collections import Counter

import numpy as np
from PIL import Image
import torch

from layer_entropy_cache import encode, make_base
from objects365_cache import WideGraphPack, block_lock, block_path, eligible_actions, split_rows
from objects365_data import BLOCK, FOLDER, HERE, INDEX, SHARDS, digest, read
from objects365_fit import ImageReplay
from objects365_instance_gate import ReceiverAttention, edge_features
from objects365_outcome_gate import state
from objects365_rare import RARE_INDEX, RARE_PLAN

OUT = FOLDER / "edge_mlp"
PLAN = HERE / "OBJECTS365_EDGE_MLP.md"


def split_document(doc, split):
    if split != "rare_train":
        return doc, split, digest(INDEX)
    rare = read(RARE_INDEX)
    assert rare["main"] == digest(INDEX) and rare["plan"] == digest(RARE_PLAN)
    return dict(doc, rows=rare["rows"]), "train", digest(RARE_INDEX)


def strict_states(frame, pack, base, doc, actions, outcomes):
    replay = ImageReplay(frame, pack, base, doc, doc["names"])
    states, owners = [], []
    for (query, proposed, _), outcome in zip(actions.tolist(), outcomes):
        level = state(outcome)
        overlaps = replay.iou[query]
        owner = -1
        if overlaps.size and overlaps.max() >= .50:
            best = int(overlaps.argmax())
            owner_class = int(replay.y[best])
            other = float(overlaps[replay.y != owner_class].max(initial=0))
            if float(overlaps[best]) - other >= .05:
                owner = owner_class
        if level >= 2 and owner != proposed:
            level = -1 if owner < 0 else 1
        states.append(level)
        owners.append(owner)
    return torch.tensor(states), torch.tensor(owners)


@torch.inference_mode()
def extract(doc, splits, shard):
    engine = WideGraphPack(doc["names"], 365, doc["parents"])
    attention = ReceiverAttention(engine)
    OUT.mkdir(parents=True, exist_ok=True)
    for split in splits:
        source_doc, original_split, index = split_document(doc,split)
        frames = {row["id"]:row for row in source_doc["rows"]}
        rows = split_rows(source_doc,original_split,shard)
        for offset in range(0,len(rows),BLOCK):
            lock = block_lock(index,rows[offset:offset+BLOCK])
            lock.update(plan=digest(PLAN), encoder=digest(HERE/"objects365_instance_gate.py"))
            output = OUT / f"{split}_{shard}_{offset//BLOCK:03d}.pt"
            if output.exists():
                assert torch.load(output,map_location="cpu",weights_only=True)["lock"] == lock
                continue
            source = torch.load(block_path(split,shard,offset//BLOCK),map_location="cpu",weights_only=True)
            assert source["lock"] == block_lock(index,rows[offset:offset+BLOCK])
            records, counts = [], Counter()
            counts["images"] = len(lock["images"])
            for record in source["records"]:
                if "actions" not in record:
                    continue
                frame = frames[record["image"]]
                actions = record["actions"]
                with Image.open(frame["path"]) as raw:
                    image = raw.convert("RGB")
                counts["actions"] += len(actions)
                for attempt in range(2):
                    pack, _, _ = engine.native_pack(image)
                    base = make_base(pack,365)
                    if split == "eval" and (not torch.equal(pack["boxes"],record["boxes"]) or
                                            not all(torch.equal(base[k],record["base"][k]) for k in base)):
                        continue
                    keys = torch.as_tensor(eligible_actions(pack,base,365)).reshape(-1,4)[:,[0,1,3]]
                    action_keys = set(map(tuple,keys.tolist()))
                    if not all(tuple(item) in action_keys for item in actions.tolist()):
                        continue
                    if split in ("policy","eval") and not torch.equal(keys,actions):
                        continue
                    old = encode(engine.reader,pack,base,actions)[:,0]
                    if float((old.float()-record["x"][:,0].float()).abs().max()) < .08:
                        break
                else:
                    counts["skipped_images"] += 1
                    counts["skipped_actions"] += len(actions)
                    print("unreproduced",split,record["image"],flush=True)
                    continue
                counts["retried"] += attempt
                queries, inverse = actions[:,0].unique(return_inverse=True)
                focus, _ = attention.crops(image,pack["boxes"][queries],queries,engine.reader)
                features = edge_features(engine.reader,pack,base,actions,focus[inverse])
                assert features.shape == (len(actions),2,3584) and features.isfinite().all()
                targets, owners = strict_states(frame,pack,base,doc,actions,record["outcomes"])
                for value,n in zip(*np.unique(targets.numpy(),return_counts=True)):
                    counts[f"target_{value}"] += int(n)
                records.append(dict(image=record["image"],actions=actions,x=features,y=targets,owner=owners))
            torch.save(dict(lock=lock,records=records,counts=dict(counts)),output.with_suffix(".part"))
            output.with_suffix(".part").replace(output)
            print(split,shard,offset//BLOCK,dict(counts),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase",choices=("fit","test"))
    parser.add_argument("--shard",type=int,required=True,choices=range(SHARDS))
    args = parser.parse_args()
    if args.phase == "test":
        receipt = read(OUT/"qualification.json")
        assert receipt["passed"] and receipt["plan"] == digest(PLAN)
    extract(read(INDEX),("train","rare_train","monitor") if args.phase=="fit" else ("policy","eval"),args.shard)
