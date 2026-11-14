"""Extract the last read token and its earlier-layer mean without donor subtraction."""
import argparse
from collections import Counter
import json

import numpy as np
from PIL import Image
import torch

from graph_residual_cache import HERE, ROOT, SHARDS, NATURAL, BLOCK, RelabelEvidence, ImageReplay
from graph_residual_cache import blocks, load, route_setup, natural_setup, sources, digest, first_index, make_base, vocabulary
from local_runtime import Runtime, clip_crops


PLAN = HERE / "LAYER_ENTROPY_PLAN_ZH.md"
FOLDER = HERE / "layer_entropy"
EXPANSIONS = (.05, .10)


def read(path):
    return json.loads(path.read_bytes())


@torch.inference_mode()
def encode(reader, pack, base, actions, crops=None):
    model = reader.graph.star
    device = model.texts.device
    positions = {int(q): i for i, q in enumerate(pack["ids"])}
    blocks = []
    for start in range(0, len(actions), 64):
        q, c, slot = actions[start:start+64].T
        assert torch.equal(base["query"][slot], q)
        roles = torch.column_stack((base["label"][slot], c)).flatten()
        rows = torch.tensor([positions[int(i)] for i in q]).repeat_interleave(2)
        text = model.texts[roles.to(device)]
        local = pack["features"][pack["i"][rows]].to(device).float()
        donor = pack["features"][pack["j"][rows]].to(device).float()
        edge = pack["stats"][rows].to(device).float()
        mask = (pack["valid"][rows] & (pack["labels"][rows] == roles[:, None]) & ~pack["high"][rows, None]).to(device)
        assert mask[1::2].any(-1).all()
        transfer = model.instance(donor) * model.embedding_scale + model.edge(edge) * model.embedding_scale + text[:, None]
        transfer = (transfer * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        read_token = model.read[None] + text
        attention = torch.ones((len(rows), 3), device=device, dtype=torch.long)
        attention[:, 1] = mask.any(1)
        views = []
        for crop in [None] + ([] if crops is None else crops):
            evidence = local if crop is None else torch.cat((local[:, :256], crop[start:start+len(q)].to(device).float().repeat_interleave(2, 0)), 1)
            seq = torch.stack((model.instance(evidence) * model.embedding_scale, transfer, read_token), dim=1)
            result = model.qwen(inputs_embeds=seq.to(model.qwen.dtype), attention_mask=attention,
                                position_ids=torch.arange(3, device=device)[None].expand(len(seq), -1),
                                output_hidden_states=True, use_cache=False)
            last = result.last_hidden_state[:, -1].float()
            earlier = torch.stack([t[:, -1].float() for t in result.hidden_states[1:-1]]).mean(0)
            assert last.shape == earlier.shape and torch.isfinite(earlier).all() and torch.isfinite(last).all()
            views.append(torch.cat((last, earlier), dim=1).reshape(len(q), -1).half().cpu())
        blocks.append(torch.stack(views, dim=1))
    return torch.cat(blocks)


def expanded(reader, path, pack, actions):
    b = pack["boxes"][actions[:, 0]].cuda()
    span = b[:, 2:] - b[:, :2]
    result = []
    with Image.open(path) as src:
        img = src.convert("RGB")
        for scale in EXPANSIONS:
            extra = scale * span
            boxes = (b + torch.cat((-extra, extra), dim=1)).clamp(0, 1)
            result.append(clip_crops(img, boxes, reader.processor, reader.clip))
    return result


def binary_labels(frame, pack, base, manifest, names, actions, stored, selected):
    replay = ImageReplay(frame, pack, base, manifest, names)
    assert torch.equal(base["query"][actions[:, 2]], actions[:, 0])
    target = torch.full((len(actions),), -1, dtype=torch.long)
    observed = stored["outcomes"][selected]
    old_loss = stored["loss"][selected] > 0
    for j, (q, c, slot) in enumerate(actions.tolist()):
        old = int(base["label"][slot])
        crowd = ((replay.ioa[q] >= .50) & np.isin(replay.cy, [old, c])).any()
        if crowd:
            continue
        iou = replay.iou[q]
        if observed[j, 1] or (iou.max(initial=0) >= .50 and iou[replay.y == c].max(initial=0) <= .30):
            target[j] = 0
        elif observed[j, 0] or old_loss[j] or stored["supports"][selected[j], 0]:
            _, lost, _, semantic, _ = replay.delta(("replace", q, c, slot), float(replay.bs[slot]))
            if any(lost) or semantic < 0:
                target[j] = 0
            elif semantic > 0:
                target[j] = 1
    return target


def save(split, shard, records, lock, counts):
    path = FOLDER / f"{split}_{shard}.pt"
    torch.save(dict(split=split, lock=lock, counts=dict(counts), records=records, eval_opened=False), path.with_suffix(".part"))
    path.with_suffix(".part").replace(path)
    print(split, shard, dict(counts), flush=True)


def historical(split, shard, reader, smoke=False):
    root, tag, _ = sources(split)
    mp = root / "manifest.json"
    manifest = read(mp)
    names, nfine, _ = vocabulary(manifest, tag)
    rows = {r["id"]: r for r in manifest["splits"][tag]}
    src = ROOT / f"receiver_local_full_{split}_{shard}.pt"
    lock = dict(plan=digest(PLAN), manifest=digest(mp), source=digest(src))
    out = FOLDER / f"{split}_{shard}.pt"
    if out.exists() and not smoke:
        previous = torch.load(out, map_location="cpu", weights_only=True)
        assert previous["lock"] == lock and not previous["eval_opened"]
        return
    data = torch.load(src, map_location="cpu", weights_only=True)
    ids = data["image"].numpy()
    counts, result = Counter(), []
    for i, iid in enumerate(np.unique(ids), 1):
        selected = np.flatnonzero(ids == iid)
        actions = torch.column_stack((data["query"][selected], data["roles"][selected, 1], data["slot"][selected]))
        pack = torch.load(root / "cache" / tag / f"{iid}.pt", map_location="cpu", weights_only=True)
        base = make_base(pack, nfine)
        target = binary_labels(rows[int(iid)], pack, base, manifest, names, actions, data, selected)
        counts.update(images=1, candidates=len(actions), positive=int((target == 1).sum()), negative=int((target == 0).sum()), unknown=int((target < 0).sum()))
        keep = target >= 0
        if keep.any():
            actions, target = actions[keep], target[keep]
            result.append(dict(image=int(iid), actions=actions, y=target, x=encode(reader, pack, base, actions)))
            if smoke and len(result) == 1:
                frame = rows[int(iid)]
                crops = expanded(reader, root / "images" / frame["dataset"] / frame["file_name"], pack, actions[:1])
                sample = encode(reader, pack, base, actions[:1], crops)
                assert sample.shape == (1, 3, 3584)
                print("view smoke", float((sample[:, 1:].float() - sample[:, :1].float()).abs().mean()), flush=True)
        if i % 48 == 0:
            print(split, shard, i, dict(counts), flush=True)
        if smoke and len(result) >= 5:
            print("hidden smoke", result[0]["x"].shape, dict(counts), flush=True)
            return
    assert result and not smoke
    save(split, shard, result, lock, counts)


def held(shard, phase, reader=None):
    result = []
    if phase == "route":
        manifest, route, _, lock = route_setup()
        names, nfine, parents = vocabulary(manifest, "cal")
        runtime = Runtime(names, nfine, parents)
        for rows, src, path in blocks(manifest, shard):
            original, admitted = load(src, rows, route["lock"]), load(path, rows, lock)
            for frame, base, item in zip(rows, original, admitted):
                chosen = base.get("selected", item)
                if "query" not in chosen:
                    continue
                act = torch.tensor([[chosen[k] for k in ("query", "proposed", "slot")]])
                ref = base["native"] if "selected" in base else item["native"]
                image_path = HERE / "images" / frame["dataset"] / frame["file_name"]
                with Image.open(image_path) as image:
                    pack, _, _ = runtime.native_pack(image.convert("RGB"))
                assert all(torch.equal(ref[k], pack[k]) for k in ("boxes", "logits"))
                x = encode(runtime.reader, pack, base["base"], act, expanded(runtime.reader, image_path, pack, act))
                result.append(dict(image=frame["id"], actions=act, x=x))
                if len(result) % 96 == 0:
                    print(phase, shard, len(result), flush=True)
    else:
        manifest, _, lock = natural_setup()
        rows = manifest["splits"]["cal_policy"][shard::SHARDS]
        for start in range(0, len(rows), BLOCK):
            archive = torch.load(NATURAL / f"natural_funnel_{shard}_{start//BLOCK:03d}.pt", map_location="cpu", weights_only=True)
            assert archive["lock"] == lock
            for frame, r in zip(rows[start:start+BLOCK], archive["records"]):
                assert frame["id"] == r["image"]
                k = r["keys"]
                if not len(k):
                    continue
                j = first_index(k[:, 0].numpy(), k[:, 1].numpy(), r["utility"].numpy())
                if r["utility"][j] <= 0:
                    continue
                act = k[j:j+1, [0, 1, 3]]
                path = NATURAL / "images" / frame["dataset"] / frame["file_name"]
                x = encode(reader, r["pack"], r["base"], act, expanded(reader, path, r["pack"], act))
                result.append(dict(image=r["image"], actions=act, x=x))
    save(phase, shard, result, dict(plan=digest(PLAN), cohort=lock), Counter(candidates=len(result)))


@torch.inference_mode()
def main(phase, shard):
    FOLDER.mkdir(exist_ok=True)
    if phase == "route":
        held(shard, phase)
        return
    manifest, _, _ = natural_setup()
    names, nfine, parents = vocabulary(manifest, "cal_policy")
    reader = RelabelEvidence()
    reader.vocabulary(names, nfine, parents)
    if phase == "smoke":
        historical("train", shard, reader, True)
    elif phase == "historical":
        for split in ("train", "mine", "cal"):
            historical(split, shard, reader)
    else:
        held(shard, phase, reader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("smoke", "historical", "route", "natural"))
    parser.add_argument("--shard", type=int, choices=range(SHARDS), default=0)
    args = parser.parse_args()
    main(args.phase, args.shard)
