"""Cache frozen receiver and donor representations for the three-arm pilot."""
import argparse
from collections import Counter
import json

import numpy as np
from PIL import Image
import torch

from route_screen import HERE, ROOT, SHARDS, save_json, vocabulary
from admission_screen import blocks, load, setup as route_setup
from competition_cache import claim_targets
from counterfactual import ImageReplay, make_base
from foreground_full_screen import digest
from local_runtime import Runtime
from natural_funnel import BLOCK, HERE as NATURAL, setup as natural_setup
from operational_cache import sources
from paired_hypotheses import RelabelEvidence, hypotheses
from receiver_admission_train import first_index


PLAN = HERE / "GRAPH_RESIDUAL_PLAN_ZH.md"
FOLDER = HERE / "graph_residual"
HISTORIC = ("train", "mine", "cal")


def labels(frame, pack, base, manifest, names, actions):
    replay = ImageReplay(frame, pack, base, manifest, names)
    q, proposed, slot = actions.T
    assert torch.equal(base["query"][slot], q)
    roles = torch.stack((base["label"][slot], proposed), dim=1)
    support = claim_targets(frame, manifest, names, replay.boxes[q], roles).numpy()
    result = []
    for j, (query, c, s) in enumerate(actions.tolist()):
        act = ("replace", query, c, s)
        new, lost, fp, sem, _ = replay.delta(act, float(replay.bs[s]))
        harm = sem < 0 or any(lost)
        observed = float(replay.iou[query, replay.y == c].max(initial=0))
        equivocal = min(support[j]) >= 0 and (support[j].sum() == 2 or (support[j, 0] == 0 and .40 <= observed < .50))
        y = 0. if harm else 1. if sem > 0 else .5 if equivocal else -1.
        result.append(dict(target=y, correct=int(sem > 0), wrong=int(sem < 0), unsupported=int(not support[j].any()),
                           lost=[len(x) for x in lost], new=[len(x) for x in new], fp=fp))
    return result


def encode(reader, pack, base, actions):
    all_fields = {k: [] for k in ("own", "donor_residual", "supported")}
    for chunk in actions.split(32):
        q, c, slot = chunk.T
        roles = torch.column_stack((base["label"][slot], c))
        assert torch.equal(base["query"][slot], q) and (roles[:, 0] != roles[:, 1]).all()
        item = hypotheses(reader, pack, q, roles)
        valid = item["supported"]
        assert valid[:, 1].all() and not item["donor_residual"][~valid].any()
        for k in all_fields:
            assert torch.isfinite(item[k]).all()
            all_fields[k].append(item[k] if k == "supported" else item[k].half())
    return {k: torch.cat(v) for k, v in all_fields.items()}


def save(split, shard, records, lock):
    counts = Counter(str(x["target"]) for r in records for x in r.get("outcomes", []))
    path = FOLDER / f"{split}_{shard}.pt"
    payload = dict(lock=lock, split=split, records=records, counts=dict(counts), eval_opened=False)
    torch.save(payload, path.with_suffix(".part"))
    path.with_suffix(".part").replace(path)
    print(split, shard, len(records), dict(counts), flush=True)


def historic(split, shard, reader, smoke=False):
    root, tag, _ = sources(split)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    names, nfine, _ = vocabulary(manifest, tag)
    assert nfine == 48
    rows = {r["id"]: r for r in manifest["splits"][tag]}
    src = ROOT / f"receiver_local_{split}_{shard}.pt"
    stored = torch.load(src, map_location="cpu", weights_only=True)
    image_ids = stored["image"].numpy()
    records = []
    for iid in np.unique(image_ids):
        indices = np.flatnonzero(image_ids == iid)
        q = stored["query"][indices].numpy()
        c = stored["roles"][indices, 1].numpy()
        order = np.lexsort((c, q, -stored["utility"][indices].numpy()))
        keep = stored["hard_fix"][indices] | stored["hard_bad"][indices]
        keep[order[:8]] = True
        selected = indices[keep.numpy()]
        actions = torch.column_stack((stored["query"][selected], stored["roles"][selected, 1], stored["slot"][selected]))
        pack = torch.load(root / "cache" / tag / f"{iid}.pt", map_location="cpu", weights_only=True)
        base = make_base(pack, nfine)
        outcomes = labels(rows[int(iid)], pack, base, manifest, names, actions)
        eligible = [j for j, item in enumerate(outcomes) if item["target"] >= 0]
        if not eligible:
            continue
        actions = actions[eligible]
        records.append(dict(image=int(iid), actions=actions, outcomes=[outcomes[j] for j in eligible],
                            **encode(reader, pack, base, actions)))
        if len(records) % 96 == 0:
            print(split, shard, len(records), flush=True)
        if smoke and len(records) == 5:
            print("smoke", records[0]["own"].shape, dict(Counter(str(x["target"]) for r in records for x in r["outcomes"])), flush=True)
            return
    assert not smoke and records
    save(split, shard, records, dict(plan=digest(PLAN), manifest=digest(manifest_path), source=digest(src)))


def natural(shard, reader):
    manifest, _, lock = natural_setup()
    rows = manifest["splits"]["cal_policy"][shard::SHARDS]
    records = []
    for start in range(0, len(rows), BLOCK):
        saved = torch.load(NATURAL / f"natural_funnel_{shard}_{start//BLOCK:03d}.pt", map_location="cpu", weights_only=True)
        assert saved["lock"] == lock
        for frame, item in zip(rows[start:start + BLOCK], saved["records"]):
            assert frame["id"] == item["image"]
            keys = item["keys"]
            if not len(keys):
                continue
            index = first_index(keys[:, 0].numpy(), keys[:, 1].numpy(), item["utility"].numpy())
            if item["utility"][index] <= 0:
                continue
            action = keys[index:index + 1, [0, 1, 3]]
            records.append(dict(image=item["image"], actions=action, **encode(reader, item["pack"], item["base"], action)))
    save("natural", shard, records, dict(plan=digest(PLAN), natural=lock))


def calibration(shard):
    manifest, route, _, lock = route_setup()
    names, nfine, parents = vocabulary(manifest, "cal")
    runtime = Runtime(names, nfine, parents)
    records = []
    for rows, src, path in blocks(manifest, shard):
        original = load(src, rows, route["lock"])
        admitted = load(path, rows, lock)
        for frame, base, item in zip(rows, original, admitted):
            chosen = base.get("selected", item)
            if "query" not in chosen:
                continue
            action = torch.tensor([[chosen[k] for k in ("query", "proposed", "slot")]])
            cached = base["native"] if "selected" in base else item["native"]
            with Image.open(HERE / "images" / frame["dataset"] / frame["file_name"]) as image:
                pack, _, _ = runtime.native_pack(image.convert("RGB"))
            assert all(torch.equal(cached[k], pack[k]) for k in ("boxes", "logits"))
            records.append(dict(image=frame["id"], actions=action, **encode(runtime.reader, pack, base["base"], action)))
            if len(records) % 96 == 0:
                print("route", shard, len(records), flush=True)
    save("route", shard, records, dict(plan=digest(PLAN), route=lock))


@torch.inference_mode()
def main(shard, phase):
    FOLDER.mkdir(exist_ok=True)
    if phase == "route":
        calibration(shard)
        return
    manifest, _, _ = natural_setup()
    names, nfine, parents = vocabulary(manifest, "cal_policy")
    reader = RelabelEvidence()
    reader.vocabulary(names, nfine, parents)
    if phase == "smoke":
        historic("train", shard, reader, True)
    elif phase == "historic":
        for split in HISTORIC:
            if not (FOLDER / f"{split}_{shard}.pt").exists():
                historic(split, shard, reader)
    else:
        natural(shard, reader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("smoke", "historic", "natural", "route"))
    parser.add_argument("--shard", type=int, choices=range(SHARDS), default=0)
    args = parser.parse_args()
    main(args.shard, args.phase)
