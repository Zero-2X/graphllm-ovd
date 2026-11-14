"""Frozen GraphLLM probe with one verified graph edge per donor view."""
import argparse
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from torchvision.ops import box_iou

from layer_entropy_cache import encode, make_base
from objects365_cache import WideGraphPack, block_lock, block_path, split_rows
from objects365_data import BLOCK, FOLDER, HERE, INDEX, SHARDS, digest, read
from objects365_fit import ImageReplay, qualified
from semantic_action_pool import eligible_actions

OUT = FOLDER / "donor_view_entropy"
PLAN = HERE / "OBJECTS365_DONOR_VIEW_ENTROPY.md"


def lock(checkpoint):
    return dict(index=digest(INDEX), plan=digest(PLAN), code=digest(Path(__file__)),
                checkpoint=checkpoint, reader=digest(HERE/"layer_entropy_cache.py"))


def paired_donors(pack, actions):
    lookup = {int(query): row for row, query in enumerate(pack["ids"])}
    rows = torch.tensor([lookup[int(query)] for query in actions[:, 0]])
    matching = pack["valid"][rows] & (pack["labels"][rows] == actions[:, 1:2])
    matching &= ~pack["high"][rows, None]
    assert ((matching.sum(1) >= 1) & (matching.sum(1) <= 2)).all()
    selected = torch.where(matching.sum(1) == 2)[0]
    rows = rows[selected]
    edges = torch.where(matching[selected])[1].reshape(-1, 2)
    nodes = pack["j"][rows[:, None], edges]
    queries = pack["node_ids"][nodes]
    distinct = (queries[:, 0] != queries[:, 1])
    if len(queries):
        first, second = pack["boxes"][queries].unbind(1)
        distinct &= box_iou(first, second).diagonal() < .50
    return selected[distinct], rows[distinct], edges[distinct], queries[distinct], len(selected)


@torch.inference_mode()
def frame_features(engine, model, doc, frame, record, counts, check):
    if "actions" not in record:
        return None
    with Image.open(frame["path"]) as image:
        pack, _, _ = engine.native_pack(image.convert("RGB"))
    base = make_base(pack, 365)
    raw = eligible_actions(pack, base, 365)
    assert raw
    actions = torch.tensor(raw)[:, [0, 1, 3]]
    assert torch.equal(actions, record["actions"]), (frame["id"], "action mismatch")
    if "base" in record:
        assert all(torch.equal(base[key], record["base"][key]) for key in base)
        assert torch.equal(pack["boxes"], record["boxes"])
    x = record["x"][:, 0].float().cuda()
    pooled = model(x).softmax(-1)[:, 1].cpu()
    if check:
        reread = encode(engine.reader, pack, base, actions)[:, 0].float().cuda()
        gap = (model(reread).softmax(-1)[:, 1].cpu()-pooled).abs().max()
        assert float(gap) <= .002, (frame["id"], float(gap))
        counts["parity_images"] += 1
        counts["parity_score_max_ppm"] = max(counts["parity_score_max_ppm"], round(float(gap)*1e6))
    selected, rows, edges, donors, pre_dedup = paired_donors(pack, actions)
    views = pooled[:, None].repeat(1, 2)
    if len(selected):
        for view in range(2):
            valid = pack["valid"].clone()
            valid[rows, edges[:, 1-view]] = False
            hidden = encode(engine.reader, dict(pack, valid=valid), base, actions[selected])[:, 0]
            views[selected, view] = model(hidden.float().cuda()).softmax(-1)[:, 1].cpu()
    assert views.isfinite().all()
    replay = ImageReplay(frame, pack, base, doc, doc["names"])
    q, c, slot = actions.T
    area = (pack["boxes"][q, 2:]-pack["boxes"][q, :2]).prod(1)
    receiver_iou = replay.iou[q.numpy()].max(axis=1, initial=0)
    donor_support = []
    for j, pair in zip(selected.tolist(), donors.tolist()):
        overlap = replay.iou[pair]
        labels = replay.y
        if overlap.shape[1]:
            best = overlap.argmax(1)
            true_donor = (overlap.max(1)>=.50) & (labels[best]==int(c[j]))
            receiver_gt = replay.iou[int(q[j])].argmax()
            true_donor &= (best!=receiver_gt) | (receiver_iou[j]<.50)
            donor_support.append(bool(true_donor.all() and best[0]!=best[1]))
        else:
            donor_support.append(False)
    counts.update(candidate_images=1, actions=len(actions), multiple_before_dedup=pre_dedup,
                  multiple=len(selected), multiple_images=int(len(selected)>0))
    return dict(image=frame["id"], actions=actions, old=base["label"][slot], views=views, pooled=pooled,
                area=area, score=base["score"][slot], receiver_iou=torch.from_numpy(receiver_iou),
                selected=selected, donors=donors, donor_support=donor_support)


def run(doc, shard, smoke):
    engine = WideGraphPack(doc["names"], 365, doc["parents"])
    model, receipt = qualified()
    anchor = lock(receipt["checkpoint"])
    for split in ("policy", "eval"):
        frames = split_rows(doc, split, shard)
        for offset in range(0, len(frames), BLOCK):
            subset = frames[offset:offset+BLOCK]
            part = offset//BLOCK
            dest = OUT / f"{split}_{shard}_{part:03d}.pt"
            source = block_path(split, shard, part)
            expected = dict(lock=anchor, source=digest(source), images=[frame["id"] for frame in subset])
            if dest.exists() and not smoke:
                assert torch.load(dest, weights_only=True)["lock"] == expected
                continue
            cache = torch.load(source, map_location="cpu", weights_only=True)
            assert cache["lock"] == block_lock(anchor["index"], subset)
            previous = {row["image"]:row for row in cache["records"]}
            results, counts = {}, Counter(images=len(subset))
            for frame in subset:
                record = previous.get(frame["id"])
                if record is None:
                    continue
                result = frame_features(engine, model, doc, frame, record, counts,
                                        check=smoke or (offset==0 and counts["candidate_images"]==0))
                if result is not None:
                    results[frame["id"]] = result
            print(dest.name, dict(counts), flush=True)
            if smoke:
                assert counts["parity_images"] >= 16 and counts["multiple"] > 0
                return
            temp = dest.with_suffix(".part")
            torch.save(dict(lock=expected, records=results, counts=dict(counts)), temp)
            temp.replace(dest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, choices=range(SHARDS), default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    run(read(INDEX), args.shard, args.smoke)

