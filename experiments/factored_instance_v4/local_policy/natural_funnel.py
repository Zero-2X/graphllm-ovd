"""Replay the frozen natural cohort and separate nomination, ranking and admission."""
import argparse
from collections import Counter
import hashlib
import json

import numpy as np
from PIL import Image
import torch

from semantic_natural import HERE, ONLINE_NAME, ROOT, SHARDS, SemanticRuntime, describe, save_json, vocabulary
from counterfactual import ImageReplay, PROTECT


BLOCK = 24


def owner_slots(base):
    q = base["query"].numpy()
    first = np.unique(q, return_index=True)[1]
    return first[base["score"].numpy()[first] < PROTECT]


def outcomes(replay, keys, scores, utility, wrong):
    if replay.iou.shape[1]:
        truth = replay.iou.argmax(1)
        nearest_iou = replay.iou[np.arange(len(truth)), truth]
        nearest = np.where(nearest_iou >= .50, replay.y[truth], -1)
    else:
        nearest = np.full(len(replay.boxes), -1, dtype=np.int64)
        nearest_iou = np.zeros(len(replay.boxes), dtype=np.float32)
    wrong = wrong[(nearest[replay.bq[wrong]] >= 0) & (nearest[replay.bq[wrong]] != replay.bc[wrong])]
    if not len(keys):
        return dict(wrong=wrong, nearest=nearest, nearest_iou=nearest_iou, correct=[], safe=[], first=None)
    q, c, _, slots = keys.T
    assert np.array_equal(replay.bq[slots], q) and np.isin(slots, owner_slots(replay.base)).all()
    assert len(np.unique(keys[:, :2], axis=0)) == len(keys) and np.isfinite(utility).all()
    nominations = np.flatnonzero((nearest[q] == c) & np.isin(slots, wrong))
    order = np.lexsort((c, q, -utility))
    top = int(order[0])
    exact = {}
    for index in np.union1d(nominations, [top]).tolist():
        action = ("replace", int(q[index]), int(c[index]), int(slots[index]))
        receipt = describe(replay, action, float(scores[slots[index]]))
        receipt["safe_fix"] = int(receipt["semantic_fix"] and not receipt["semantic_bad"] and not receipt["lost_floor"])
        exact[index] = receipt
    assert not exact[top]["semantic_fix"] or top in nominations
    correct = [i for i in nominations.tolist() if exact[i]["semantic_fix"]]
    safe = [i for i in correct if exact[i]["safe_fix"]]
    return dict(wrong=wrong, nearest=nearest, nearest_iou=nearest_iou, nominations=nominations,
                correct=correct, safe=safe, first=top, first_outcome=exact[top])


def record_summary(keys, utility, result, historical, threshold):
    brief = Counter(images=1, candidates=len(keys), wrong_slots=len(result["wrong"]),
                    wrong_images=int(len(result["wrong"]) > 0))
    assert len(keys) == historical["candidates"]
    if not len(keys):
        assert "utility" not in historical
        return dict(brief), 0.
    top = result["first"]
    nom = result["nominations"]
    correct = result["correct"]
    safe = result["safe"]
    brief.update(correct_nominations=len(nom), nominated_slots=len(np.unique(keys[nom, 3])),
                 nominated_images=int(len(nom) > 0), actual_actions=len(correct),
                 actual_slots=len(np.unique(keys[correct, 3])), actual_images=int(bool(correct)),
                 safe_actions=len(safe), safe_slots=len(np.unique(keys[safe, 3])), safe_images=int(bool(safe)),
                 first_correct=int(top in correct), first_safe=int(top in safe),
                 positive_first_correct=int(utility[top] > 0 and top in correct),
                 admitted_correct=int(utility[top] >= threshold and top in correct),
                 correct_at_2_to_5=0, correct_at_6_or_lower=0)
    if correct and top not in correct:
        rank = np.lexsort((keys[:, 1], keys[:, 0], -utility))
        first_correct = int(np.flatnonzero(np.isin(rank, correct))[0]) + 1
        brief["correct_at_2_to_5" if first_correct <= 5 else "correct_at_6_or_lower"] += 1
    if "utility" not in historical:
        assert utility[top] <= 0
        return dict(brief), 0.
    q, c, d, _ = map(int, keys[top])
    assert (q, c, d) == (historical["query"], historical["proposed"], historical["donor"])
    assert utility[top] > 0 and (utility[top] >= threshold) == (historical["utility"] >= threshold)
    error = abs(float(utility[top]) - historical["utility"])
    assert error <= .0001, (historical["image"], error)
    assert all(historical["outcome"][k] == v for k, v in result["first_outcome"].items())
    return dict(brief), error


def setup():
    path = HERE / "manifest.json"
    manifest = json.loads(path.read_text())
    prior = json.loads((HERE / "semantic_natural_result.json").read_text())
    assert prior["passed"] and not prior["eval_opened"]
    lock = dict(manifest=hashlib.sha256(path.read_bytes()).hexdigest(),
                model=hashlib.sha256((ROOT / f"{ONLINE_NAME}_best.pt").read_bytes()).hexdigest(),
                plan=hashlib.sha256((HERE / "NATURAL_FUNNEL_PLAN_ZH.md").read_bytes()).hexdigest(),
                threshold=prior["best"]["threshold"])
    return manifest, prior, lock


@torch.inference_mode()
def run(shard):
    manifest, _, lock = setup()
    names, nfine, parents = vocabulary(manifest, "cal_policy")
    assert nfine == 48
    baseline = json.loads((HERE / f"semantic_natural_{shard}.json").read_text())
    rows = manifest["splits"]["cal_policy"][shard::SHARDS]
    assert [row["id"] for row in rows] == [r["image"] for r in baseline]
    runtime = SemanticRuntime(names, nfine, parents)
    for start in range(0, len(rows), BLOCK):
        frames = rows[start:start + BLOCK]
        path = HERE / f"natural_funnel_{shard}_{start // BLOCK:03d}.pt"
        if path.exists():
            saved = torch.load(path, map_location="cpu", weights_only=True)
            assert saved["lock"] == lock and [x["image"] for x in saved["records"]] == [r["id"] for r in frames]
            continue
        records = []
        for row, historical in zip(frames, baseline[start:start + BLOCK]):
            with Image.open(HERE / "images" / row["dataset"] / row["file_name"]) as source:
                image = source.convert("RGB")
            pack, base, keys, x, utility, _ = runtime.encode(image)
            replay = ImageReplay(row, pack, base, manifest, names)
            result = outcomes(replay, keys, base["score"].numpy(), utility, owner_slots(base))
            brief, error = record_summary(keys, utility, result, historical, lock["threshold"])
            records.append(dict(image=row["id"], pack=pack, base=base, keys=torch.from_numpy(keys),
                                evidence=None if x is None else torch.from_numpy(x),
                                utility=None if utility is None else torch.from_numpy(utility),
                                brief=brief, max_utility_error=error))
        torch.save(dict(lock=lock, records=records), path.with_suffix(".part"))
        path.with_suffix(".part").replace(path)
        print(shard, start + len(frames), len(rows), flush=True)


def finish():
    manifest, prior, lock = setup()
    totals_new = Counter()
    max_error = 0.
    for shard in range(SHARDS):
        rows = manifest["splits"]["cal_policy"][shard::SHARDS]
        for start in range(0, len(rows), BLOCK):
            path = HERE / f"natural_funnel_{shard}_{start // BLOCK:03d}.pt"
            data = torch.load(path, map_location="cpu", weights_only=True)
            assert data["lock"] == lock and [x["image"] for x in data["records"]] == [r["id"] for r in rows[start:start + BLOCK]]
            for record in data["records"]:
                totals_new.update(record["brief"])
                max_error = max(max_error, record["max_utility_error"])
    assert totals_new["images"] == prior["images"] and totals_new["candidates"] == prior["candidates"]
    assert totals_new["positive_first_correct"] == prior["all_winners"]["semantic_fix"]
    assert totals_new["admitted_correct"] == prior["selected"]["semantic_fix"]
    funnel = dict(wrong_images=totals_new["wrong_images"], no_nomination=totals_new["wrong_images"]-totals_new["nominated_images"],
                  not_executable=totals_new["nominated_images"]-totals_new["actual_images"],
                  not_first=totals_new["actual_images"]-totals_new["first_correct"],
                  nonpositive_first=totals_new["first_correct"]-totals_new["positive_first_correct"],
                  rejected_positive_first=totals_new["positive_first_correct"]-totals_new["admitted_correct"],
                  adopted=totals_new["admitted_correct"])
    assert all(x >= 0 for x in funnel.values()) and sum(v for k, v in funnel.items() if k != "wrong_images") == funnel["wrong_images"]
    frontier = []
    for budget in (0, 1, 2):
        for preserve in (False, True):
            eligible = [r for r in prior["thresholds"] if r["semantic_bad"] <= budget and (not preserve or not r["lost_floor"])]
            best = max(eligible, key=lambda r: (r["semantic_fix"], -r["semantic_bad"], -r["lost_floor"], -r["edits"]))
            point = {k: best[k] for k in ("semantic_fix", "semantic_bad", "lost_floor", "edits", "threshold")}
            frontier.append(dict(budget=budget, no_loss_required=preserve,
                                 net=point["semantic_fix"] - point["semantic_bad"], **point))
    summary = dict(lock=lock, verified=dict(totals_new), funnel=funnel, old_frontier=frontier,
                   max_utility_error=max_error, eval_opened=False)
    save_json(HERE / "natural_funnel_result.json", summary)
    print(summary, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--shard", type=int, choices=range(SHARDS))
    mode.add_argument("--finish", action="store_true")
    args = parser.parse_args()
    if args.finish:
        finish()
    else:
        run(args.shard)
