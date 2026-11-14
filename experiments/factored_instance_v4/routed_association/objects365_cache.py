"""Extract the frozen 365-class detector graph and confirmed action features."""
import argparse
from collections import Counter
from pathlib import Path

from PIL import Image
import torch

from layer_entropy_cache import encode, expanded, make_base
from donor_transport_probe import Native
from objects365_data import BLOCK, FOLDER, INDEX, PLAN, SHARDS, digest, read, save_json
from objects365_rare import RARE_INDEX, RARE_PLAN
from omni_layer_cache import GraphPack, adjudicate, eligible_actions


def text_capacity(native, tokens):
    original = native.model.config.max_text_len
    maximum = max(original, tokens)
    native.model.config.max_text_len = maximum
    native.model.model.config.max_text_len = maximum
    heads = [layer for layer in native.model.modules() if type(layer).__name__ == "GroundingDinoContrastiveEmbedding"]
    assert len(heads) >= 2
    for head in heads:
        head.max_text_len = maximum


class WideGraphPack(GraphPack):
    def __init__(self, names, nfine, parents):
        super().__init__(names, nfine, parents)
        self.tokens = len(self.native.ids)
        assert self.tokens > self.native.model.config.max_text_len and nfine == 365
        text_capacity(self.native, self.tokens)


def split_rows(doc, split, shard):
    return [row for row in doc["rows"] if row["split"] == split][shard::SHARDS]


def block_path(split, shard, part):
    return FOLDER / f"{split}_{shard}_{part:03d}.pt"


def block_lock(index_hash, rows):
    return dict(index=index_hash, images=[r["id"] for r in rows])


@torch.inference_mode()
def extract(frame, engine, doc, split, counts):
    source = Path(frame["path"])
    assert digest(source) == frame["sha"]
    with Image.open(source) as image:
        pack, _, _ = engine.native_pack(image.convert("RGB"))
    assert pack["logits"].shape == (900, len(doc["names"])) and torch.isfinite(pack["logits"]).all()
    base = make_base(pack, 365)
    keys = eligible_actions(pack, base, 365)
    counts["images"] += 1
    counts["candidates"] += len(keys)
    record = dict(image=frame["id"], split=split)
    if split == "eval":
        record.update(base=base, boxes=pack["boxes"])
    if not keys:
        return record if split == "eval" else None
    actions = torch.tensor(keys)[:, [0, 1, 3]]
    outcomes = adjudicate(frame, pack, base, doc, actions)
    counts.update(str(o["target"]) for o in outcomes)
    keep = [i for i, o in enumerate(outcomes) if o["target"] >= 0 or split in ("policy", "eval")]
    if not keep:
        return None
    actions = actions[keep]
    views = expanded(engine.reader, source, pack, actions) if split in ("policy", "eval") else None
    features = encode(engine.reader, pack, base, actions, views)
    assert features.shape == (len(actions), 3 if views is not None else 1, 3584)
    record.update(actions=actions, x=features, outcomes=[outcomes[i] for i in keep])
    return record


@torch.inference_mode()
def smoke(doc):
    frame = next(r for r in doc["rows"] if r["split"] == "train")
    with Image.open(frame["path"]) as source:
        image = source.convert("RGB")
    native = Native(doc["base"][:48])
    before, scores_before, _, _ = native.forward(image)
    text_capacity(native, 1024)
    after, scores_after, _, _ = native.forward(image)
    assert torch.equal(before, after) and torch.equal(scores_before, scores_after)
    del native
    torch.cuda.empty_cache()
    engine = WideGraphPack(doc["names"], 365, doc["parents"])
    counts = Counter()
    for row in (r for r in doc["rows"] if r["split"] == "train"):
        extract(row, engine, doc, "train", counts)
        if counts["images"] == 16:
            break
    result = dict(short_equal=True, tokens=engine.tokens, fine=365, counts=dict(counts),
                  peak_gib=torch.cuda.max_memory_allocated()/2**30)
    save_json(FOLDER / "smoke.json", result)
    print("Objects365 smoke", result, flush=True)


@torch.inference_mode()
def run(doc, shard, splits, rare=False):
    index_hash = digest(RARE_INDEX if rare else INDEX)
    engine = WideGraphPack(doc["names"], 365, doc["parents"])
    for split in splits:
        rows = split_rows(doc, split, shard)
        for start in range(0, len(rows), BLOCK):
            group = rows[start:start+BLOCK]
            path = block_path("rare_train" if rare else split, shard, start//BLOCK)
            lock = block_lock(index_hash, group)
            if path.exists():
                assert torch.load(path, map_location="cpu", weights_only=True)["lock"] == lock
                continue
            counts, records = Counter(), []
            for frame in group:
                record = extract(frame, engine, doc, split, counts)
                if record is not None:
                    records.append(record)
            torch.save(dict(lock=lock, records=records, counts=dict(counts)), path.with_suffix(".part"))
            path.with_suffix(".part").replace(path)
            print(split, shard, start//BLOCK, dict(counts), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("smoke", "train", "rare", "eval"))
    parser.add_argument("--shard", type=int, default=0)
    args = parser.parse_args()
    doc = read(INDEX)
    assert doc["plan"] == digest(PLAN) and doc["official_split"] == "train"
    if args.mode == "smoke":
        smoke(doc)
    elif args.mode == "train":
        assert read(FOLDER / "smoke.json")["short_equal"]
        run(doc, args.shard, ("train", "monitor", "policy"))
    elif args.mode == "rare":
        rare = read(RARE_INDEX)
        assert rare["main"] == digest(INDEX) and rare["plan"] == digest(RARE_PLAN)
        run(dict(doc, rows=rare["rows"]), args.shard, ("train",), rare=True)
    else:
        repair = FOLDER / "qualification_repair.json"
        receipt = read(repair if repair.exists() else FOLDER / "qualification.json")
        assert receipt["passed"] and receipt["index"] == digest(INDEX)
        assert (FOLDER / "policy.json").exists()
        run(doc, args.shard, ("eval",))
