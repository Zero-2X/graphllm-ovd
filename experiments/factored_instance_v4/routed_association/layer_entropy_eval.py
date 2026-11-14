"""Calibrate mean support with and without mixed-entropy rejection."""
import numpy as np
import torch

from graph_residual_eval import check_action, detector_metrics, frontier, passed, point_summary, route_outcomes
from layer_entropy_cache import FOLDER, HERE, PLAN, SHARDS, digest, read
from layer_entropy_train import qualified
from ownership_graph_audit import auc
from route_screen import save_json


def entropy(p):
    p = p.clamp(1e-7, 1-1e-7)
    return -p*p.log()-(1-p)*(1-p).log()


@torch.inference_mode()
def scores(model, split, doc):
    entries, hashes = {}, {}
    for shard in range(SHARDS):
        path = FOLDER / f"{split}_{shard}.pt"
        cached = torch.load(path, map_location="cpu", weights_only=True)
        assert cached["split"] == split and cached["lock"]["plan"] == doc["plan"] and not cached["eval_opened"]
        records = cached["records"]
        x = torch.cat([r["x"] for r in records]).float().cuda()
        assert x.ndim == 3 and x.shape[1] == 3
        p = model(x.flatten(0, 1)).softmax(1)[:, 1].reshape(-1, 3)
        mean = p.mean(1)
        avg_entropy = entropy(p).mean(1)
        total_entropy = entropy(mean)
        divergence = (total_entropy - avg_entropy).clamp_min(0)
        for r, views, value, intrinsic, total, spread in zip(records, p.tolist(), mean.tolist(), avg_entropy.tolist(), total_entropy.tolist(), divergence.tolist()):
            q, c, slot = r["actions"][0].tolist()
            entries[r["image"]] = dict(image=r["image"], query=q, proposed=c, slot=slot, views=views, score=value,
                                       within_entropy=intrinsic, total_entropy=total, js=spread)
        hashes[path.name] = digest(path)
    save_json(FOLDER / f"{split}_scores.json", dict(training=doc["receipt"], inputs=hashes, rows=entries, labels_read=False))
    return entries


def eligible(rows, cap=None):
    return [r for r in rows if r["score"] > .5 and (cap is None or r["js"] <= cap)]


def main():
    model, doc = qualified()
    cal = route_outcomes(scores(model, "route", doc))
    good = [r["js"] for r in cal if r["target"] == 1. and r["views"][0] > .5]
    assert good
    cap = float(np.percentile(good, 90))
    rules = dict(mean=None, mixed_entropy=cap)
    points = {k: frontier(eligible(cal, v), [], (0,), True)[0] for k,v in rules.items()}
    save_json(FOLDER / "calibration.json", dict(training=doc["receipt"], divergence_limit=cap, thresholds=points,
                                                reference_count=len(good), natural_outcomes_read=False, eval_opened=False))
    print("mixed entropy thresholds", cap, points, flush=True)
    natural = scores(model, "natural", doc)
    old = read(HERE / "receiver_admission_v3_result.json")
    energy = read(HERE / "energy_admission_result.json")
    foreground = read(HERE / "foreground_full_lock.json")
    baseline = read(HERE / "foreground_admission_full_result.json")["foreground_calibrated"][0]["threshold"]
    anchors = {r["image"] for r in old["anchors"]} | {r["image"] for r in foreground["natural"] if r["score"] >= baseline}
    anchors |= set(energy["energy"]["ids"])
    rows = []
    for prior in old["anchors"] + old["eligible"]:
        evidence = natural[prior["image"]]
        check_action(evidence, prior)
        rows.append(dict(prior, **{k:v for k,v in evidence.items() if k not in ("image","query","proposed","slot")}))
    assert len(rows) == len(natural) == 945 and sum(r["correct"] for r in rows) == 157
    selected = {k: [r for r in eligible(rows, v) if passed(r["score"], points[k]["threshold"])] for k,v in rules.items()}
    diagnostics = [r for r in cal if r["target"] in (0., 1.)]
    y = np.array([r["target"] for r in diagnostics])
    discrimination = {k: auc(np.array([r[k] if k == "score" else -r[k] for r in diagnostics]), y)
                      for k in ("score", "total_entropy", "within_entropy", "js")}
    result = dict(calibration=points, cap=cap, natural={k:point_summary(v, anchors) for k,v in selected.items()},
                  frontier={k:frontier(eligible(rows, v), [], (0,1,2), True) for k,v in rules.items()},
                  ap=detector_metrics(selected), hard_diagnostic_auc=discrimination,
                  distinct_views=sum(len(set(r["views"]))>1 for r in rows), eval_opened=False)
    save_json(FOLDER / "result.json", result)
    print("mixed entropy result", result, flush=True)


if __name__ == "__main__":
    main()
