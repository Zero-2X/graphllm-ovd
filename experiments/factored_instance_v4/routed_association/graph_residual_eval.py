"""Lock scores and evaluate actual original-score relabeling."""
import argparse
import json

import numpy as np
import torch

from admission_eval import frontier, passed, summarize
from admission_screen import blocks, load, setup as route_setup
from foreground_full_eval import detector_metrics
from graph_residual_cache import FOLDER, HERE, PLAN, SHARDS, digest, labels, save_json, vocabulary
from graph_residual_train import ARMS, PCA_RECEIPT, RECEIPT, qualified, raw


def read(path):
    return json.loads(path.read_bytes())


@torch.inference_mode()
def scores(models, doc, split, receipt):
    items = []
    inputs = {}
    for shard in range(SHARDS):
        path = FOLDER / f"{split}_{shard}.pt"
        cached = torch.load(path, map_location="cpu", weights_only=True)
        assert cached["split"] == split and cached["lock"]["plan"] == doc["plan"] and not cached["eval_opened"]
        assert all("outcomes" not in r for r in cached["records"])
        items.extend(cached["records"])
        inputs[path.name] = digest(path)
    x, hidden = raw(items)
    assert hidden == doc["hidden"] and len(x) == len(items) == len({r["image"] for r in items})
    predictions = {arm: models[arm](x.cuda()).softmax(-1)[:, 1].tolist() for arm in ARMS}
    entries = {}
    for i, row in enumerate(items):
        q, c, slot = row["actions"][0].tolist()
        entries[row["image"]] = dict(image=row["image"], query=q, proposed=c, slot=slot,
                                      scores={arm: predictions[arm][i] for arm in ARMS})
    suffix = "_pca" if doc.get("rank") else ""
    save_json(FOLDER / f"{split}_scores{suffix}.json", dict(plan=doc["plan"], qualification=digest(receipt),
                                                    inputs=inputs, rows=entries, labels_read=False))
    return entries


def check_action(row, expected):
    assert all(row[k] == expected[k] for k in ("query", "proposed", "slot")), row["image"]


def route_outcomes(entries):
    manifest, route, _, lock = route_setup()
    names, nfine, _ = vocabulary(manifest, "cal")
    assert nfine == 48
    annotations = read(HERE / "base_annotations.json")
    outcomes = []
    for shard in range(SHARDS):
        for rows, src, path in blocks(manifest, shard):
            original = load(src, rows, route["lock"])
            admitted = load(path, rows, lock)
            for frame, base, item in zip(rows, original, admitted):
                chosen = base.get("selected", item)
                if "query" not in chosen:
                    continue
                row = entries[frame["id"]]
                check_action(row, chosen)
                pack = base["native"] if "selected" in base else item["native"]
                row_labels = labels(dict(frame, annotations=annotations[str(frame["id"])]), pack, base["base"],
                                    manifest, names, torch.tensor([[row[k] for k in ("query", "proposed", "slot")]]))
                outcomes.append(dict(row, **row_labels[0]))
    assert len(entries) == len(outcomes) == 2889
    return outcomes


def candidates(rows, arm):
    return [dict(r, score=r["scores"][arm]) for r in rows if r["scores"][arm] > .5]


def point_summary(selected, anchors):
    retained = sum(r["correct"] and r["image"] in anchors for r in selected)
    return dict(**summarize(selected), retained=retained)


def main(pca=False):
    models, doc = qualified(pca)
    receipt = PCA_RECEIPT if pca else RECEIPT
    suffix = "_pca" if pca else ""
    route = route_outcomes(scores(models, doc, "route", receipt))
    calibration = {arm: frontier(candidates(route, arm), [], (0,), True)[0] for arm in ARMS}
    save_json(FOLDER / f"calibration{suffix}.json", dict(plan=digest(PLAN), training=digest(receipt), thresholds=calibration,
                                                natural_outcomes_read=False, eval_opened=False))
    print("graph residual thresholds", calibration, flush=True)

    natural = scores(models, doc, "natural", receipt)
    prior = read(HERE / "receiver_admission_v3_result.json")
    frozen = read(HERE / "foreground_full_lock.json")
    old = read(HERE / "foreground_admission_full_result.json")
    energy = read(HERE / "energy_admission_result.json")
    assert all(not r["eval_opened"] for r in (prior, frozen, old, energy))
    cutoff = old["foreground_calibrated"][0]["threshold"]
    current = {r["image"] for r in prior["anchors"]} | {r["image"] for r in frozen["natural"] if r["score"] >= cutoff}
    current |= set(energy["energy"]["ids"])
    rows = []
    for known in prior["anchors"] + prior["eligible"]:
        evidence = natural[known["image"]]
        check_action(evidence, known)
        rows.append(dict(known, scores=evidence["scores"]))
    assert len(natural) == len(rows) == 945 and sum(r["correct"] for r in rows) == 157
    assert summarize([r for r in rows if r["image"] in current]) == energy["energy"]["total"]

    locked, dev = {}, {}
    for arm in ARMS:
        choices = candidates(rows, arm)
        locked[arm] = [r for r in choices if passed(r["score"], calibration[arm]["threshold"])]
        dev[arm] = frontier(choices, [], (0, 1, 2), True)
    metrics = detector_metrics(locked)
    metrics.update(read(HERE / "energy_admission_metrics.json")["ap"])
    stats = {}
    for arm in ARMS:
        stats[arm] = {str(value): float(np.mean([r["scores"][arm] for r in route if r["target"] == value]))
                      for value in (0., .5, 1.)}
    result = dict(calibration=calibration, locked={k: point_summary(v, current) for k, v in locked.items()},
                  same_risk_development=dev, ap=metrics, baseline=energy["energy"]["total"],
                  mean_support_on_route_labels=stats, eval_opened=False)
    save_json(FOLDER / f"result{suffix}.json", result)
    print("graph residual evaluation", result, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pca", action="store_true")
    main(parser.parse_args().pca)
