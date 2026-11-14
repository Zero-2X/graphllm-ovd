"""Retrieve whole-image seeds from actual donor regions on opened development images."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torchvision.ops import box_iou, nms, roi_align

from proposal_smoke import EXTENT, ProposalEngine, V1, actual_donors, save_json, vocabulary, xyxy
from donor_transport_probe import SIZE, transport
from counterfactual import gt_arrays, make_base


ROOT = Path(__file__).resolve().parent
SHARDS = 4
COLUMNS = (
    "image", "query", "class", "strict", "old_harmed", "lost_floor", "aligned",
    "native_support", "seed_overlap", "decoded_overlap", "native_class_overlap",
    "truth_support", "utility",
)


def load_actions():
    rows = np.concatenate([np.load(ROOT/f"visual_pair_{i}.npz")["rows"] for i in range(SHARDS)])
    assert len(rows) == 1964 and rows[:,4:6].sum(0).tolist() == [10, 114]
    index = defaultdict(list)
    for row in rows:
        index[int(row[0])].append(row)
    return index


def patches(field, scale, boxes):
    patch = roi_align(field, [boxes*scale], SIZE, sampling_ratio=2, aligned=True)
    return F.normalize(patch.flatten(2).transpose(1,2), dim=-1)


def retrieve(engine, args, field, scale, coord, anchors, valid, donor_box):
    allowed = valid & (box_iou(anchors, donor_box).flatten() < .30)
    indices = torch.nonzero(allowed).flatten()
    support = patches(field, scale, donor_box)
    scores = []
    for part in indices.split(512):
        candidate = patches(field, scale, anchors[part])
        sim = support @ candidate.transpose(1, 2)
        score, _ = transport(sim)
        scores.append(score)
    seeds = indices[nms(anchors[indices], torch.cat(scores), .50)[:16]]
    assert len(seeds) == 16
    boxes, prob = engine.decode_seeds(args, coord[seeds])
    distinct = box_iou(boxes, donor_box).flatten() < .30
    return anchors[seeds], boxes[distinct], prob[distinct]


def check_action(q, c, old, d, receiver, seeds, boxes, prob, base_boxes, base_prob, truth, truth_labels):
    seed_overlap = float(box_iou(seeds, receiver).max())
    decoded_overlap = class_overlap = truth_support = utility = 0.
    if len(boxes):
        overlap = box_iou(boxes, receiver).flatten()
        confidence, leaf = prob.max(1)
        proposed = (leaf == c) & (confidence >= .05)
        qualified = proposed & (overlap >= .50)
        decoded_overlap = float(overlap.max())
        class_overlap = float(overlap.masked_fill(~proposed, 0).max())
        margin = (prob[:, c] - prob[:, old]).clamp_min(0)
        utility = float((overlap * confidence * margin).masked_fill(~qualified, 0).max())
        if qualified.any() and len(truth):
            # Audit complete same-class instances separately from receiver overlap.
            donor_hit = box_iou(base_boxes[d:d+1], truth).flatten()
            receiver_hit = box_iou(receiver, truth).flatten()
            matching_truth = (truth_labels == c) & (donor_hit < .50) & (receiver_hit >= .50)
            if matching_truth.any():
                truth_support = float(box_iou(boxes[qualified], truth[matching_truth]).max())
    native_confidence, native_leaf = base_prob.max(1)
    native_overlap = box_iou(base_boxes, receiver).flatten()
    other = (native_leaf == c) & (native_confidence >= .05) & (native_overlap >= .50)
    other[q] = other[d] = False
    aligned = bool(len(truth) and float(box_iou(receiver, truth).max()) >= .50)
    return (int(aligned), int(other.any()), seed_overlap, decoded_overlap, class_overlap, truth_support, utility)


@torch.inference_mode()
def run(shard):
    index = load_actions()
    manifest = json.loads((V1/"manifest.json").read_text())
    names, nfine, _ = vocabulary(manifest, "cal")
    images = [row for row in manifest["splits"]["cal"] if row["id"] in index][shard::SHARDS]
    engine = ProposalEngine(names, nfine)
    records = []
    for i, row in enumerate(images, 1):
        iid = row["id"]
        pack = torch.load(V1/"cache/cal"/(str(iid)+".pt"), map_location="cpu", weights_only=True)
        base = make_base(pack, nfine)
        old_by_query = {}
        for query, label in zip(base["query"].tolist(), base["label"].tolist()):
            old_by_query.setdefault(query, label)
        image = Image.open(V1/"images"/row["dataset"]/row["file_name"]).convert("RGB")
        base_boxes, logits, field, scale = engine.native.forward(image)
        assert torch.equal(base_boxes.cpu(), pack["boxes"])
        assert torch.equal(logits.cpu(), pack["logits"])
        raw = engine.native.last_output
        args = dict(engine.decoder_args)
        coord = raw.enc_outputs_coord_logits[0]
        anchors = xyxy(coord.sigmoid())
        valid = torch.isfinite(coord).all(1) & ((anchors[:,2:]-anchors[:,:2])>0).all(1)
        truth, labels = gt_arrays(row, names, manifest)
        truth, labels = truth.cuda(), labels.cuda()
        base_prob = logits[:,:nfine].sigmoid()
        donor = actual_donors(pack, nfine)
        decoded = {}
        for action in index[iid]:
            _, q, c = map(int, action[:3])
            d = donor[q,c]
            if d not in decoded:
                decoded[d] = retrieve(engine, args, field, scale, coord, anchors, valid, base_boxes[d:d+1])
            values = check_action(q, c, old_by_query[q], d, base_boxes[q:q+1], *decoded[d],
                                  base_boxes, base_prob, truth, labels)
            records.append((iid, q, c, *action[4:7], *values))
        if i % 5 == 0 or i == 1 or i == len(images):
            print("dense donor", shard, i, len(images), len(records), flush=True)
    path = ROOT/f"dense_donor_{shard}.npz"
    with path.with_suffix(".part").open("wb") as out:
        np.savez_compressed(out, rows=np.asarray(records), columns=np.array(COLUMNS))
    path.with_suffix(".part").replace(path)


def report():
    rows = np.concatenate([np.load(ROOT/f"dense_donor_{i}.npz")["rows"] for i in range(SHARDS)])
    rows = rows[np.lexsort((rows[:,2], rows[:,1], rows[:,0]))]
    assert np.isfinite(rows).all() and len(rows) == len(np.unique(rows[:,:3], axis=0)) == 1964
    strict, harm, loss, aligned = (rows[:,j].astype(bool) for j in range(3, 7))
    native = rows[:,7] > 0
    seed, decoded, proposed, truth = (rows[:,j]>=.50 for j in range(8,12))
    utility = rows[:,12]
    assert (strict.sum(), harm.sum(), (~aligned).sum()) == (10,114,1760)
    assert ((strict&native).sum(), (harm&native).sum(), (~aligned&native).sum()) == (4,7,119)

    def counts(mask):
        return dict(candidates=int(mask.sum()), strict=int((mask&strict).sum()),
                    old_harmed=int((mask&harm).sum()), lost_floor=int((mask&loss).sum()),
                    unaligned=int((mask&~aligned).sum()))

    groups = {}
    for name, mask in dict(strict=strict, old_harmed=harm, unaligned=~aligned).items():
        groups[name] = dict(rows=int(mask.sum()), native=int((mask&native).sum()),
                            seeded=int((mask&seed).sum()), decoded=int((mask&decoded).sum()),
                            proposed=int((mask&proposed).sum()), verified_truth=int((mask&truth).sum()),
                            new_support=int((mask&proposed&~native).sum()))
    best = {}
    for j in np.flatnonzero(utility>0):
        iid = int(rows[j,0])
        if iid not in best or utility[j]>utility[best[iid]]:
            best[iid] = j
    rank = np.array(sorted(best.values(), key=lambda j:-utility[j]), dtype=int)
    maximum = 0
    qualified = None
    good = damage = lost = 0
    for i, j in enumerate(rank, 1):
        good += int(strict[j])
        damage += int(harm[j])
        lost += int(loss[j])
        if i < len(rank) and utility[rank[i]] == utility[j]:
            continue
        if damage or lost:
            break
        maximum = max(maximum, good)
        if good>=2 and good/i>=.50:
            qualified = dict(actions=i, strict=good, threshold=float(utility[j]))
    result = dict(scope="previously opened development actions", groups=groups,
                  accepted=counts(utility>0), winners=counts(np.isin(np.arange(len(rows)),rank)),
                  max_strict_without_harm=maximum, qualified_prefix=qualified, passed=qualified is not None,
                  strict_rows=rows[strict].tolist(), columns=list(COLUMNS))
    save_json(ROOT/"dense_donor_result.json", result)
    print(json.dumps({k:v for k,v in result.items() if k not in ("strict_rows","columns")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, choices=range(SHARDS))
    args = parser.parse_args()
    if args.shard is None:
        report()
    else:
        run(args.shard)
