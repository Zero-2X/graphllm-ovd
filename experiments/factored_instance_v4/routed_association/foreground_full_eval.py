"""Calibrate and evaluate foreground replacement on complete locked cohorts."""
import json

import torch

from admission_eval import calibrate, frontier, natural_setup, BLOCK, summarize, passed
from foreground_full_screen import HERE, NATURAL, RESULT, SHARDS, digest, setup
from receiver_admission_train import NAME as PREVIOUS
from route_screen import vocabulary, save_json
from counterfactual import predictions, ImageReplay
from evaluate import coco_eval, metrics


NAME = "foreground_admission_full"


def detector_metrics(picks):
    manifest, _, lock = natural_setup()
    names, nfine, _ = vocabulary(manifest, "cal_policy")
    frames = manifest["splits"]["cal_policy"]
    catids = [manifest["categories"][x] for x in names[:nfine]]
    chosen = {k: {r["image"]: r for r in rows} for k, rows in picks.items()}
    output = {k: [] for k in picks}
    seen = set()
    for shard in range(SHARDS):
        rows = frames[shard::SHARDS]
        for start in range(0, len(rows), BLOCK):
            data = torch.load(NATURAL / f"natural_funnel_{shard}_{start // BLOCK:03d}.pt", map_location="cpu", weights_only=True)
            assert data["lock"] == lock
            for row, record in zip(rows[start:start + BLOCK], data["records"]):
                iid = row["id"]
                assert iid == record["image"] and iid not in seen
                seen.add(iid)
                base = record["base"]
                original = predictions(row, base, catids)
                replay = None
                for mode in picks:
                    a = chosen[mode].get(iid)
                    if a is None:
                        output[mode].extend(original)
                        continue
                    if replay is None:
                        replay = ImageReplay(row, record["pack"], base, manifest, names)
                    slot = a["slot"]
                    changed = replay.output(("replace", a["query"], a["proposed"], slot), float(base["score"][slot]))
                    output[mode].extend(predictions(row, changed, catids))
    assert len(seen) == len(frames) == 1320
    return {k: metrics(coco_eval(frames, v, catids)) for k, v in output.items()}


@torch.inference_mode()
def main():
    manifest, route, _, prior_lock, _, _, _, lock = setup()
    full = json.loads(RESULT.read_bytes())
    assert full["lock"] == lock and not full["labels_used"] and not full["eval_opened"]
    cal_scores = {r["image"]: r["score"] for r in full["calibration"]}
    assert len(cal_scores) == len(full["calibration"]) == 2850
    cal = calibrate(manifest, route, prior_lock, cal_scores)
    cal.update(foreground_lock=digest(RESULT))
    save_json(HERE / f"{NAME}_calibration.json", cal)
    print("foreground thresholds locked", cal, flush=True)

    original = json.loads((HERE / f"{PREVIOUS}_result.json").read_bytes())
    assert original["calibration"]["lock"] == prior_lock and not original["eval_opened"]
    anchors, previous = original["anchors"], original["eligible"]
    scores = {r["image"]: r for r in full["natural"]}
    assert len(scores) == len(full["natural"]) == len(previous) == 925
    assert all(r["score"] == scores[r["image"]]["prior"] for r in previous)
    changed = [dict(r, score=scores[r["image"]]["score"]) for r in previous]
    old_cut = original["locked"][0]["threshold"]
    baseline = [r for r in previous if passed(r["score"], old_cut)]
    kept_before = {r["image"] for r in baseline if r["correct"]}
    assert len(kept_before) == 11 and len(anchors) == 20
    def collect(rows, cut):
        additional = [r for r in rows if passed(r["score"], cut)]
        fixed = {r["image"] for r in additional if r["correct"]}
        return additional, dict(threshold=cut, **summarize(anchors + additional), added=summarize(additional),
                                previous_correct_kept=len(fixed & kept_before), newly_correct=len(fixed - kept_before))
    transferred, old_threshold = collect(changed, old_cut)
    calibrated = [collect(changed, point["threshold"]) for point in cal["thresholds"]]
    out = detector_metrics(dict(foreground_old_cut=anchors + transferred, foreground_calibrated=anchors + calibrated[0][0]))
    out.update(native=original["ap"]["native"], old_direct=original["ap"]["old_direct"],
               previous_admission=original["ap"]["admission_locked_zero"])
    _, previous_summary = collect(previous, old_cut)
    report = dict(previous=previous_summary, foreground_old_cut=old_threshold,
                  foreground_calibrated=[x[1] for x in calibrated], calibration=cal,
                  zero_loss_frontier=frontier(changed, anchors, (0, 1, 2), True), ap=out,
                  old_score_max_error=full["old_score_max_error"], eval_opened=False)
    save_json(HERE / f"{NAME}_result.json", report)
    print(report, flush=True)


if __name__ == "__main__":
    main()
