"""Calibrate locked natural scores and evaluate fixed-first admission."""
from collections import Counter
import hashlib
from itertools import groupby
import json

import torch

from admission_screen import HERE, RESULT, SHARDS, SPLIT, blocks, load, setup
from natural_funnel import HERE as NATURAL, BLOCK, setup as natural_setup
from route_screen import save_json, vocabulary
from receiver_admission_train import NAME, THRESHOLD, eligible, first_index, frozen, outcome, single_features
from counterfactual import ImageReplay, predictions
from evaluate import coco_eval, metrics


def counts(rows):
    result = Counter(actions=len(rows))
    for row in rows:
        for name in ("correct", "wrong", "unsupported"):
            result[name] += row[name]
        for field in ("lost", "new", "fp"):
            for j, val in enumerate(row[field]):
                result[f"{field}_{j}"] += val
    return result


def summarize(rows):
    return format_counts(counts(rows))


def format_counts(values):
    return dict(actions=values["actions"], correct=values["correct"], wrong=values["wrong"],
                net=values["correct"] - values["wrong"], unsupported=values["unsupported"],
                **{k: [values[f"{k}_{j}"] for j in range(6)] for k in ("lost", "new", "fp")})


def frontier(picks, anchors, budgets, preserve):
    initial = counts(anchors)
    current = initial.copy()
    best = [dict(threshold=None, **format_counts(initial)) for _ in budgets]
    def objective(record):
        return record["correct"], -record["wrong"], -sum(record["lost"]), -record["actions"]
    for cutoff, tie in groupby(sorted(picks, key=lambda r: -r["score"]), key=lambda r: r["score"]):
        current.update(counts(list(tie)))
        point = format_counts(current)
        for j, budget in enumerate(budgets):
            if point["wrong"] <= budget and (not preserve or not any(point["lost"])):
                if objective(point) > objective(best[j]):
                    best[j] = dict(threshold=cutoff, **point)
    return best


def passed(score, threshold):
    return threshold is not None and score >= threshold


def calibrate(manifest, route, lock, replacement_scores=None):
    cache = json.loads(RESULT.read_bytes())
    assert cache["lock"] == lock and not cache["labels_read"] and not cache["eval_opened"]
    assert cache["images"] == 4096 and len(cache["chunk_sha256"]) == 128
    annotations = json.loads((HERE / "base_annotations.json").read_bytes())
    names, nfine, _ = vocabulary(manifest, "cal")
    assert nfine == 48
    picks = []
    seen = set()
    for shard in range(SHARDS):
        for rows, source, path in blocks(manifest, shard):
            assert hashlib.sha256(path.read_bytes()).hexdigest() == cache["chunk_sha256"][path.name]
            assert hashlib.sha256(source.read_bytes()).hexdigest() == route["chunk_sha256"][source.name]
            for row, record, original in zip(rows, load(path, rows, lock), load(source, rows, route["lock"])):
                iid = row["id"]
                assert iid not in seen and ("selected" not in original or "score" not in record)
                seen.add(iid)
                if "score" not in record:
                    continue
                row["annotations"] = annotations[str(iid)]
                q, c, slot = (record[k] for k in ("query", "proposed", "slot"))
                o = outcome(row, record["native"], original["base"], manifest, names, q, c, slot)
                score = record["score"] if replacement_scores is None else replacement_scores[iid]
                picks.append(dict(image=iid, score=score, **o))
    assert len(seen) == len(manifest["splits"][SPLIT]) == 4096 and len(picks) == cache["eligible"]
    if replacement_scores is not None:
        assert len(replacement_scores) == len(picks)
    budgets = (0, 3, 6)
    return dict(lock=lock, calibration_images=len(seen), eligible=len(picks), budgets=budgets,
                state_counts=dict(Counter(r["label"] for r in picks)), thresholds=frontier(picks, [], budgets, True),
                natural_read=False, eval_opened=False)


def evaluate(head, calibration):
    manifest, prior, lock = natural_setup()
    names, nfine, _ = vocabulary(manifest, "cal_policy")
    v2, reader, v2_names, prior_sha = frozen()
    assert names[:nfine] == v2_names and calibration["lock"]["prior"] == prior_sha and lock["threshold"] == THRESHOLD
    catids = [manifest["categories"][n] for n in names[:nfine]]
    anchors, extras = [], []
    output = {k: [] for k in ("native", "old_direct", "admission_locked_zero")}
    primary = calibration["thresholds"][0]["threshold"]
    frames = manifest["splits"]["cal_policy"]
    for shard in range(SHARDS):
        rows = frames[shard::SHARDS]
        for start in range(0, len(rows), BLOCK):
            path = NATURAL / f"natural_funnel_{shard}_{start // BLOCK:03d}.pt"
            data = torch.load(path, map_location="cpu", weights_only=True)
            assert data["lock"] == lock
            for row, record in zip(rows[start:start + BLOCK], data["records"]):
                assert row["id"] == record["image"]
                pack, base, keys = (record[k] for k in ("pack", "base", "keys"))
                old, new = base, base
                if len(keys):
                    utility = record["utility"].numpy()
                    k = first_index(keys[:, 0].numpy(), keys[:, 1].numpy(), utility)
                    q, c, _, slot = map(int, keys[k])
                    u = float(utility[k])
                    if u >= THRESHOLD or eligible(u):
                        state = outcome(row, pack, base, manifest, names, q, c, slot)
                        action = ("replace", q, c, slot)
                        receipt = dict(image=row["id"], query=q, proposed=c, slot=slot, old_utility=u, **state)
                        if u >= THRESHOLD:
                            anchors.append(receipt)
                            old = ImageReplay(row, pack, base, manifest, names).output(action, float(base["score"][slot]))
                            new = old
                        else:
                            x = single_features(v2, reader, pack, base, q, c, slot)
                            score = float(head.score(x[None])[0])
                            extras.append(dict(receipt, score=score))
                            if passed(score, primary):
                                new = ImageReplay(row, pack, base, manifest, names).output(action, float(base["score"][slot]))
                for name, box in zip(output, (base, old, new)):
                    output[name].extend(predictions(row, box, catids))
    baseline = summarize(anchors)
    assert baseline["actions"] == 20 and baseline["correct"] == 16 and baseline["wrong"] == baseline["lost"][0] == 0
    assert len(extras) == prior["all_winners"]["edits"] - 20
    actual = []
    for point in calibration["thresholds"]:
        cutoff = point["threshold"]
        adopted = [r for r in extras if passed(r["score"], cutoff)]
        actual.append(dict(threshold=cutoff, **summarize(anchors + adopted), added=summarize(adopted)))
    summary = dict(baseline=baseline, locked=actual, frontier=frontier(extras, anchors, (0, 1, 2), False),
                   zero_loss_frontier=frontier(extras, anchors, (0, 1, 2), True),
                   ap={k: metrics(coco_eval(frames, p, catids)) for k, p in output.items()},
                   anchors=anchors, eligible=extras, images=len(frames))
    return summary


@torch.inference_mode()
def main():
    manifest, route, head, lock = setup()
    calibration = calibrate(manifest, route, lock)
    save_json(HERE / f"{NAME}_calibration.json", calibration)
    print("locked thresholds", calibration, flush=True)
    result = evaluate(head, calibration)
    result.update(calibration=calibration, eval_opened=False)
    save_json(HERE / f"{NAME}_result.json", result)
    print({k: v for k, v in result.items() if k not in ("anchors", "eligible", "calibration")}, flush=True)


if __name__ == "__main__":
    main()
