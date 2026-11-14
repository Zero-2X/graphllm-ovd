"""Compare one-object and two-object explanations supported by other native queries."""
import argparse
import json

import numpy as np
from PIL import Image
import torch
from torchvision.ops import box_iou

from dense_donor import ROOT, SHARDS, save_json, vocabulary
from identity_features import FootprintNative, pair_features
from identity_train import IdentityReader
from operational_cache import sources
from train_factor import by_image


def hypotheses(assignment, separation):
    count = assignment.shape[1]
    if not count:
        return (0., -1), (0., -1, -1)
    common, common_at = assignment.amin(0).max(0)
    preference = assignment[0] - assignment[1]
    split = torch.minimum(torch.minimum(assignment[0, :, None], assignment[1, None, :]), separation)
    split = split.masked_fill(~((preference[:, None] > 0) & (preference[None, :] < 0)), 0)
    distinct, distinct_at = split.flatten().max(0)
    a, b = divmod(int(distinct_at), count)
    return (float(common), int(common_at) if common > 0 else -1), (float(distinct), a, b)


def score_edges(reader, native, boxes, logits, q, r, c):
    chunks = []
    for a, b in zip(np.array_split(q, max(1, (len(q) + 511) // 512)),
                    np.array_split(r, max(1, (len(q) + 511) // 512))):
        x = pair_features(native, boxes, logits, a, b, np.full(len(a), c))
        chunks.append(reader.probability(torch.from_numpy(x).cuda()))
    return torch.cat(chunks)


def candidate_bank(boxes, logits, query, c, start=0):
    iou = box_iou(boxes[query], boxes)
    selected = (iou.amax(0) > .5) & (logits[:, c].sigmoid() >= .05)
    selected[:start] = False
    selected[query] = False
    return torch.nonzero(selected).flatten().cpu().numpy()


def score_pair(reader, native, boxes, logits, query, c, conditional=False):
    pair = score_edges(reader, native, boxes, logits, query[:1], query[1:], c)[0]
    direct = float(pair.prod())
    bank = candidate_bank(boxes, logits, query, c)
    count = len(bank)
    if not count:
        return dict(candidates=0, direct=direct, common=0., distinct=0., score=0.)
    edge = score_edges(reader, native, boxes, logits, np.repeat(query, count), np.tile(bank, 2), c)
    assignment = (edge[:, 1] if conditional else edge.prod(1)).reshape(2, count)
    difference = assignment[0] - assignment[1]
    a, b = torch.nonzero((difference[:, None] > 0) & (difference[None, :] < 0), as_tuple=True)
    separation = assignment.new_zeros(count, count)
    if len(a):
        left, right = bank[a.cpu().numpy()], bank[b.cpu().numpy()]
        x = score_edges(reader, native, boxes, logits, left, right, c)
        separation[a, b] = (1 - x[:, 1]) if conditional else x[:, 0] * (1 - x[:, 1])
    common, distinct = hypotheses(assignment, separation)
    reverse = hypotheses(assignment.flip(0), separation.T)
    assert (common[0], distinct[0]) == (reverse[0][0], reverse[1][0])
    witness = bank[list(distinct[1:])]
    split_iou = float(box_iou(boxes[witness[:1]], boxes[witness[1:]])[0, 0]) if distinct[0] else None
    return dict(candidates=count, direct=direct, common=common[0], distinct=distinct[0], score=common[0] - distinct[0],
                common_query=int(bank[common[1]]) if common[1] >= 0 else None,
                distinct_queries=witness.tolist() if distinct[0] else [], distinct_box_iou=split_iou)


def screen(cases, field, abstain):
    same = np.array([x["state"] == 1 for x in cases])
    hard = np.array([x["state"] == 2 and x["box_iou"] >= .7 for x in cases])
    scores = np.array([x[field] for x in cases])
    contrasts = scores[same, None] - scores[~same]
    eligible = scores > 0 if abstain else np.ones(len(cases), dtype=bool)
    positives = np.sort(scores[same & eligible])
    accepted = (scores >= positives[-17]) & eligible if len(positives) >= 17 else None
    false_at17 = int((accepted & ~same).sum()) if accepted is not None else None
    hard_at17 = int((accepted & hard).sum()) if accepted is not None else None
    return dict(auc=float(((contrasts > 0) + .5 * (contrasts == 0)).mean()), eligible_same=len(positives),
                positive_before_first_wrong=int((same & eligible & (scores > scores[~same].max())).sum()),
                different_at17=false_at17, hard_at17=hard_at17,
                passed=false_at17 is not None and false_at17 <= 5 and hard_at17 == 0)


@torch.inference_mode()
def main(conditional):
    mode = "conditional" if conditional else "witness"
    assert (ROOT / f"OWNERSHIP_{mode.upper()}_PLAN_ZH.md").is_file()
    if conditional:
        trace = json.loads((ROOT / "ownership_witness_trace.json").read_text())
        exposed = [c for c in trace["cases"] if c["query"] == [241, 5]][0]["groups"][0]
        assert exposed["correct_product_preference"] == 0 and exposed["correct_conditional_preference"] == 6
    prior = json.loads((ROOT / "ownership_witness_supply.json").read_text())
    assert prior["hard"]["both_pre_nms"] == 3
    receipt = json.loads((ROOT / "identity_qualification.json").read_text())
    checkpoint = torch.load(ROOT / "identity_best.pt", map_location="cpu", weights_only=True)
    assert receipt["passed"] and checkpoint["epoch"] == receipt["best_epoch"]
    reader = IdentityReader().cuda().eval()
    reader.load_state_dict(checkpoint["model"])
    old = np.concatenate([np.load(ROOT / f"joint_affine_probe_{i}.npz")["rows"] for i in range(SHARDS)])
    index = by_image(old)
    source, tag, _ = sources("cal")
    manifest = json.loads((source / "manifest.json").read_text())
    names, nfine, _ = vocabulary(manifest, tag)
    assert nfine == 48
    native = FootprintNative(names)
    cases = []
    for row in manifest["splits"][tag]:
        iid = row["id"]
        if iid not in index:
            continue
        image = Image.open(source / "images" / row["dataset"] / row["file_name"]).convert("RGB")
        boxes, logits, _, _ = native.forward(image)
        pack = torch.load(source / "cache" / tag / f"{iid}.pt", map_location="cpu", weights_only=True)
        assert torch.equal(boxes.cpu(), pack["boxes"]) and torch.equal(logits.cpu(), pack["logits"])
        for pair in index[iid]:
            q, c = pair[[1, 4]].astype(int), int(pair[2])
            x = score_pair(reader, native, boxes, logits, q, c, conditional)
            cases.append(dict(image=iid, query=q.tolist(), state=int(pair[7]), box_iou=float(pair[12]), **x))
        if len(cases) % 10 == 0:
            print("whole-instance evidence", len(cases), flush=True)
    assert len(cases) == 110 and len(index) == 77
    split = [r for r in cases if r.get("distinct_box_iou") is not None]
    result = dict(pairs=len(cases), images=len(index), direct=screen(cases, "direct", False),
                  witness=screen(cases, "score", True), split_supported=len(split),
                  split_outside_edge_training=sum(r["distinct_box_iou"] <= .5 for r in split),
                  hard=[r for r in cases if r["state"] == 2 and r["box_iou"] >= .7],
                  scope="frozen hypotheses on the fixed 110 development pairs")
    save_json(ROOT / f"ownership_{mode}_result.json", dict(result, cases=cases))
    print(result, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditional", action="store_true")
    main(parser.parse_args().conditional)
