"""Run the frozen GraphLLM admission policy on image-only cohorts."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "local_policy"))
from semantic_natural import ONLINE_NAME, ROOT, SemanticRuntime, save_json, vocabulary

SHARDS = 4
BLOCK = 32
THRESHOLD = 0.9368263483047485


def setup():
    manifest_path = HERE / "routing_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    fetch = json.loads((HERE / "routing_fetch_receipt.json").read_text())
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert fetch["hash_disjoint"] and fetch["images"] == 8192 and fetch["manifest_sha256"] == manifest_sha
    original = HERE.parent / "local_policy"
    assert json.loads((original / "semantic_natural_smoke.json").read_text())["passed"]
    prior = json.loads((original / "semantic_natural_result.json").read_text())
    assert prior["passed"] and not prior["eval_opened"] and prior["best"]["threshold"] == THRESHOLD
    assert not (HERE / "base_annotations.json").exists()
    assert all("annotations" not in r for rows in manifest["splits"].values() for r in rows)
    model_path = ROOT / f"{ONLINE_NAME}_best.pt"
    lock = dict(manifest=manifest_sha, model=hashlib.sha256(model_path.read_bytes()).hexdigest(), threshold=THRESHOLD)
    return manifest, lock


def screen(runtime, row):
    path = HERE / "images" / row["dataset"] / row["file_name"]
    with Image.open(path) as stream:
        image = stream.convert("RGB")
    pack, base, keys, _, score, coverage = runtime.encode(image)
    assert pack["boxes"].shape == (900, 4) and len(pack["logits"]) == 900
    assert all(len(v) == len(base["query"]) for v in base.values())
    result = dict(image=int(row["id"]), candidates=len(keys), base={k: v.cpu() for k, v in base.items()})
    if len(keys):
        assert np.isfinite(score).all()
        ix = int(np.lexsort((keys[:, 1], keys[:, 0], -score))[0])
        if score[ix] >= THRESHOLD:
            q, c, d, slot = map(int, keys[ix])
            assert int(base["query"][slot]) == q and int(base["label"][slot]) != c
            other, same = map(bool, coverage[ix])
            result["selected"] = dict(query=q, proposed=c, donor=d, slot=slot, utility=float(score[ix]),
                                      other_native=other, same_query=same)
            result["native"] = {k: pack[k].cpu() for k in ("boxes", "logits")}
    return result


def chunks(manifest, shard):
    for name, rows in manifest["splits"].items():
        local = rows[shard::SHARDS]
        for start in range(0, len(local), BLOCK):
            yield name, local[start:start + BLOCK], HERE / f"screen_{name}_{shard}_{start // BLOCK:03d}.pt"


def load(path, rows, lock):
    data = torch.load(path, map_location="cpu", weights_only=True)
    assert data["lock"] == lock
    assert [x["image"] for x in data["records"]] == [r["id"] for r in rows]
    return data["records"]


@torch.inference_mode()
def run(shard, smoke):
    manifest, lock = setup()
    if not smoke:
        assert json.loads((HERE / "routing_smoke.json").read_text())["lock"] == lock
    names, nfine, parents = vocabulary(manifest, "cal")
    assert nfine == 48
    runtime = SemanticRuntime(names, nfine, parents)
    if smoke:
        picked = [r for rows in manifest["splits"].values() for r in rows[:2]]
        trace = [screen(runtime, r) for r in picked]
        assert len(trace) == 4 and sum(r["candidates"] for r in trace) > 0
        result = dict(lock=lock, images=len(trace), candidates=sum(r["candidates"] for r in trace),
                      selected=sum("selected" in r for r in trace), passed=True)
        save_json(HERE / "routing_smoke.json", result)
        print(result, flush=True)
        return
    total = {k: 0 for k in manifest["splits"]}
    selected = total.copy()
    for split, rows, path in chunks(manifest, shard):
        if path.exists():
            records = load(path, rows, lock)
        else:
            records = [screen(runtime, r) for r in rows]
            temp = path.with_suffix(".part")
            torch.save(dict(lock=lock, records=records), temp)
            temp.replace(path)
        total[split] += len(records)
        selected[split] += sum("selected" in x for x in records)
        print("routed screen", shard, split, total[split], selected[split], flush=True)


def lock_results():
    manifest, lock = setup()
    counts = {k: dict(images=0, candidates=0, selected=0, with_reference=0) for k in manifest["splits"]}
    admitted, hashes, observed = [], {}, set()
    for shard in range(SHARDS):
        for split, rows, path in chunks(manifest, shard):
            records = load(path, rows, lock)
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            for row in records:
                iid = row["image"]
                assert iid not in observed
                observed.add(iid)
                counts[split]["images"] += 1
                counts[split]["candidates"] += row["candidates"]
                if "selected" in row:
                    x = row["selected"]
                    assert x["utility"] >= THRESHOLD
                    counts[split]["selected"] += 1
                    counts[split]["with_reference"] += int(x["other_native"] or x["same_query"])
                    admitted.append(dict(split=split, image=iid, **x))
    assert len(observed) == 8192 and len(admitted) == sum(r["selected"] for r in counts.values())
    result = dict(lock=lock, splits=counts, selection=admitted, chunk_sha256=hashes, labels_read=False, eval_opened=False)
    save_json(HERE / "routing_selection.json", result)
    print(counts, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--shard", type=int, choices=range(SHARDS))
    modes.add_argument("--smoke", action="store_true")
    modes.add_argument("--lock", action="store_true")
    args = parser.parse_args()
    if args.lock:
        lock_results()
    else:
        run(args.shard, args.smoke)
