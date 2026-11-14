"""Fit and evaluate the Objects365-only cross-layer admission head."""
import argparse
from collections import Counter

import numpy as np
import torch

from objects365_cache import block_lock, block_path, split_rows
from admission_eval import frontier, passed, summarize
from counterfactual import ImageReplay, predictions
from evaluate import coco_eval, metrics
from layer_entropy_eval import entropy
from objects365_data import BLOCK, FOLDER, HERE, INDEX, PLAN, SHARDS, digest, read, save_json
from objects365_rare import RARE_INDEX, RARE_PLAN
from omni_layer_train import Head, EPOCHS, SEED, losses, tensorize, train_epoch
from ownership_graph_audit import auc


RECEIPT = FOLDER / "qualification.json"
REPAIR_PLAN = HERE / "OBJECTS365_REPAIR.md"


def blocks(doc, split, rare=False):
    index_hash = digest(RARE_INDEX if rare else INDEX)
    by_id = {r["id"]:r["split"] for r in doc["rows"]}
    for shard in range(SHARDS):
        rows = split_rows(doc, split, shard)
        for start in range(0, len(rows), BLOCK):
            path = block_path("rare_train" if rare else split, shard, start//BLOCK)
            cache = torch.load(path, map_location="cpu", weights_only=True)
            assert cache["lock"] == block_lock(index_hash, rows[start:start+BLOCK])
            assert all(r["split"] == split and by_id[r["image"]] == split for r in cache["records"])
            yield cache


def collect(doc, split, rare=False):
    rows, counts = [], Counter()
    for block in blocks(doc, split, rare):
        rows.extend(block["records"])
        counts.update(block["counts"])
    assert counts["images"] == doc["counts"][split]
    return rows, dict(counts)


def checkpoint(model, epoch, name):
    path = FOLDER / f"{name}.pt"
    torch.save(dict(model=model.state_dict(), epoch=epoch, plan=digest(PLAN)), path.with_suffix(".part"))
    path.with_suffix(".part").replace(path)


def fit(doc, repair=False):
    torch.manual_seed(SEED)
    rows, main_counts = collect(doc, "train")
    rare = read(RARE_INDEX)
    assert rare["main"] == digest(INDEX) and rare["plan"] == digest(RARE_PLAN)
    additional, rare_counts = collect(dict(rows=rare["rows"], counts=dict(train=len(rare["rows"]))), "train", rare=True)
    rows.extend(additional)
    train_counts = dict(Counter(main_counts) + Counter(rare_counts))
    monitor_rows, monitor_counts = collect(doc, "monitor")
    training_ids = {r["image"] for r in rows}
    assert not training_ids.intersection(r["image"] for r in monitor_rows)
    classes = Counter(int(action[1]) for r in rows for action, item in zip(r["actions"], r["outcomes"])
                      if item["target"] == 1)
    receipt = dict(train=train_counts, monitor=monitor_counts, train_labeled_images=len(rows),
                   main_train=main_counts, rare_train=rare_counts, annotated_train_classes=rare["annotated_train_classes"],
                   rare_index=digest(RARE_INDEX), rare_plan=digest(RARE_PLAN),
                   positive_classes=len(classes), nonperson_positive=sum(n for c,n in classes.items() if doc["base"][c] != "person"),
                   train_positive_images=sum(any(o["target"]==1 for o in r["outcomes"]) for r in rows),
                   plan=digest(PLAN), index=digest(INDEX), eval_opened=False)
    save_json(FOLDER / "supply.json", receipt)
    print("Objects365 supervision", receipt, flush=True)
    train, monitor = tensorize(rows), tensorize(monitor_rows)
    model = Head().cuda()
    if repair:
        assert not (FOLDER / "qualification_repair.json").exists()
        previous = read(RECEIPT)
        assert not previous["passed"] and previous["index"] == digest(INDEX) and previous["rare_index"] == digest(RARE_INDEX) and not previous["eval_opened"]
        assert digest(FOLDER/"best.pt") == previous["checkpoint"]
        model.load_state_dict(torch.load(FOLDER/"best.pt", map_location="cpu", weights_only=True)["model"])
        receipt.update(repair_plan=digest(REPAIR_PLAN), original_best_epoch=previous["best_epoch"],
                       original_best=previous["best"], original_final=previous["final"])
        suffix, epochs, lr, average_start = "_repair", 16, .00001, 1
    else:
        assert not RECEIPT.exists()
        suffix, epochs, lr, average_start = "", EPOCHS, .0005, 25
    average = torch.optim.swa_utils.AveragedModel(model)
    if repair:
        average.update_parameters(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.03)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    initial = losses(model, monitor)
    boot = np.random.default_rng(SEED).integers(0, len(initial), (2000, len(initial)))
    low, high = np.percentile(initial[boot].mean(1), [2.5, 97.5])
    tolerance = previous["tolerance"] if repair else max(.005, float((high-low)/2))
    best, best_epoch = (float(initial.mean()), 0) if repair else (float("inf"), -1)
    if repair:
        checkpoint(average.module, 0, "best"+suffix)
    history = []
    for epoch in range(1, epochs+1):
        train_epoch(model, train, optimizer)
        schedule.step()
        if epoch >= average_start:
            average.update_parameters(model)
        tracked = average.module if epoch >= average_start else model
        valid = float(losses(tracked, monitor).mean())
        train_value = float(losses(tracked, train).mean())
        if valid < best:
            best, best_epoch = valid, epoch
            checkpoint(tracked, epoch, "best"+suffix)
        history.append(dict(epoch=epoch, cal=valid, train=train_value, lr=schedule.get_last_lr()[0]))
        save_json(FOLDER / f"history{suffix}.json", history)
        print(history[-1], flush=True)
    checkpoint(average.module, epochs, "final"+suffix)
    tail_cal = np.array([r["cal"] for r in history[-5:]])
    tail_train = np.array([r["train"] for r in history[-5:]])
    plateau = abs(tail_train[:2].mean()-tail_train[-2:].mean())/max(1e-8, tail_train[:2].mean())
    slope = abs(np.polyfit(np.arange(5), tail_cal, 1)[0])
    checks = dict(schedule=history[-1]["lr"]==0., finite=True, plateau=bool(plateau<=.025),
                  slope=bool(slope<=tolerance/4), spread=bool(np.ptp(tail_cal)<=tolerance),
                  gap=bool(tail_cal[-1]-best<=tolerance))
    receipt.update(passed=all(checks.values()), checks=checks, best=best, best_epoch=best_epoch, initial=float(initial.mean()),
                   final=float(tail_cal[-1]), tolerance=tolerance, plateau=float(plateau), slope=float(slope),
                   parameters=sum(p.numel() for p in model.parameters()), model="best"+suffix+".pt",
                   checkpoint=digest(FOLDER/("best"+suffix+".pt")), repair=repair)
    save_json(FOLDER / f"qualification{suffix}.json", receipt)
    print("Objects365 qualification", receipt, flush=True)


def qualified():
    path = FOLDER / "qualification_repair.json"
    if not path.exists():
        path = RECEIPT
    receipt = read(path)
    assert receipt["passed"] and receipt["plan"] == digest(PLAN) and receipt["index"] == digest(INDEX)
    assert receipt["rare_index"] == digest(RARE_INDEX) and receipt["rare_plan"] == digest(RARE_PLAN)
    if receipt["repair"]:
        assert receipt["repair_plan"] == digest(REPAIR_PLAN)
    assert receipt["checkpoint"] == digest(FOLDER / receipt["model"])
    state = torch.load(FOLDER / receipt["model"], map_location="cpu", weights_only=True)
    assert state["plan"] == receipt["plan"] and state["epoch"] == receipt["best_epoch"]
    model = Head().cuda().eval()
    model.load_state_dict(state["model"])
    return model, receipt


@torch.inference_mode()
def first_actions(records, model):
    rows = []
    for record in records:
        if "actions" not in record:
            continue
        x = record["x"].float().cuda()
        p = model(x.flatten(0, 1)).softmax(-1)[:, 1].reshape(-1, 3)
        average = p.mean(1)
        best = int(average.argmax())
        js = (entropy(average[best])-entropy(p[best]).mean()).clamp_min(0)
        q, c, slot = record["actions"][best].tolist()
        rows.append(dict(record["outcomes"][best], image=record["image"], query=q, proposed=c, slot=slot,
                         score=float(average[best]), js=float(js)))
    return rows


def options(rows, cap):
    return [r for r in rows if r["score"] > .5 and (cap is None or r["js"] <= cap)]


def calibrate(doc):
    model, receipt = qualified()
    records, counts = collect(doc, "policy")
    rows = first_actions(records, model)
    observed = [r for r in rows if r["target"] >= 0]
    correct = [r["js"] for r in observed if r["correct"] and r["score"] > .5]
    cap = float(np.percentile(correct,90)) if correct else 0.
    cutoffs = {k:frontier(options(observed, v), [], (0,), True)[0]
               for k, v in (("mean", None), ("mixed_entropy", cap))}
    unconfirmed = {k:sum(r["target"]<0 and passed(r["score"],cutoffs[k]["threshold"]) for r in options(rows,v))
                   for k,v in (("mean", None), ("mixed_entropy", cap))}
    result = dict(checkpoint=receipt["checkpoint"], cap=cap, cap_positive=len(correct), cutoffs=cutoffs,
                  images=counts["images"], first=len(rows), confirmed=len(observed), candidate_counts=counts,
                  confirmed_correct=sum(r["correct"] for r in observed), unconfirmed=unconfirmed, eval_opened=False)
    save_json(FOLDER / "policy.json", result)
    print("Objects365 policy", result, flush=True)


def replay_metrics(doc, records, selected):
    by_image = {k:{r["image"]:r for r in actions} for k,actions in selected.items()}
    pred = {k:[] for k in ("native", *selected)}
    frames, annotations = [], []
    cat_ids = [doc["categories"][name] for name in doc["base"]]
    for record in records:
        frame = doc["rows"][record["image"]]
        frames.append(frame)
        annotations.extend(dict(a, image_id=frame["id"], area=a["bbox"][2]*a["bbox"][3])
                           for a in frame["annotations"])
        base = record["base"]
        pred["native"].extend(predictions(frame, base, cat_ids))
        for k in selected:
            row = by_image[k].get(frame["id"])
            if row is None:
                pred[k].extend(predictions(frame, base, cat_ids))
                continue
            replay = ImageReplay(frame, record, base, doc, doc["names"])
            action = ("replace", row["query"], row["proposed"], row["slot"])
            pred[k].extend(predictions(frame, replay.output(action, float(base["score"][row["slot"]])), cat_ids))
    assert len(frames) == len(records)
    for i, annotation in enumerate(annotations,1):
        annotation["id"] = i
    ap = {k:metrics(coco_eval(frames, value, cat_ids, annotations)) for k,value in pred.items()}
    return ap, len({a["category_id"] for a in annotations})


def evaluate(doc):
    model, receipt = qualified()
    policy = read(FOLDER / "policy.json")
    assert policy["checkpoint"] == receipt["checkpoint"]
    records, counts = collect(doc, "eval")
    rows = first_actions(records, model)
    selected = {k:[r for r in options(rows, cap) if passed(r["score"],policy["cutoffs"][k]["threshold"])]
                for k, cap in (("mean", None), ("mixed_entropy", policy["cap"]))}
    ap, annotated_classes = replay_metrics(doc, records, selected)
    diagnostic = {k:frontier(options(rows, cap), [], (0, 1), True)
                  for k, cap in (("mean", None), ("mixed_entropy", policy["cap"]))}
    observed = [r for r in rows if r["target"]>=0]
    correct = np.array([bool(r["correct"]) for r in observed])
    result = dict(images=len(records), annotated_classes=annotated_classes,
                  candidate_counts=counts, first=len(rows), first_correct=sum(r["correct"] for r in rows),
                  possible_images=sum(any(o["correct"] for o in r.get("outcomes", [])) for r in records),
                  support_auc=auc(np.array([r["score"] for r in observed]),correct),
                  low_disagreement_auc=auc(-np.array([r["js"] for r in observed]),correct),
                  policy=policy, counts={k:summarize(value) for k,value in selected.items()}, ap=ap,
                  unconfirmed={k:sum(r["target"]<0 for r in value) for k,value in selected.items()},
                  frontier=diagnostic, eval_opened=True)
    save_json(FOLDER / "result.json", result)
    print("Objects365 result", result, flush=True)


@torch.inference_mode()
def coco_transfer():
    from graph_residual_eval import check_action, detector_metrics
    from layer_entropy_cache import FOLDER as COCO_FEATURES, SHARDS as COCO_SHARDS
    from objects365_data import HERE
    from omni_layer_eval import score

    model, receipt = qualified()
    policy = read(FOLDER / "policy.json")
    assert policy["checkpoint"] == receipt["checkpoint"]
    records = []
    for shard in range(COCO_SHARDS):
        cache = torch.load(COCO_FEATURES/f"natural_{shard}.pt", map_location="cpu", weights_only=True)
        assert not cache["eval_opened"] and all("outcomes" not in r for r in cache["records"])
        records.extend(cache["records"])
    scored = {r["image"]: r for r in score(records, model)}
    previous = read(HERE / "receiver_admission_v3_result.json")
    rows = []
    for item in previous["anchors"]+previous["eligible"]:
        prediction = scored[item["image"]]
        check_action(prediction, item)
        rows.append(dict(item, score=prediction["score"], js=prediction["js"]))
    assert len(rows) == len(scored) == 945 and sum(r["correct"] for r in rows) == 157
    choices = {k:options(rows, cap) for k, cap in (("mean", None), ("mixed_entropy", policy["cap"]))}
    selected = {k:[r for r in values if passed(r["score"],policy["cutoffs"][k]["threshold"])]
                for k,values in choices.items()}
    observed = [r for r in rows if r["correct"] or r["wrong"] or any(r["lost"])]
    target = np.array([bool(r["correct"]) for r in observed])
    result = dict(images=1320, candidates=len(rows), first_correct=157, policy=digest(FOLDER/"policy.json"),
                  counts={k:summarize(values) for k,values in selected.items()}, ap=detector_metrics(selected),
                  support_auc=auc(np.array([r["score"] for r in observed]),target),
                  low_disagreement_auc=auc(-np.array([r["js"] for r in observed]),target),
                  frontier={k:frontier(values,[],(0,1),True) for k,values in choices.items()}, sealed_eval_opened=False)
    save_json(FOLDER / "coco_transfer.json", result)
    print("Objects365 COCO transfer", result, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("train", "repair", "policy", "eval", "coco"))
    args = parser.parse_args()
    doc = read(INDEX)
    assert doc["plan"] == digest(PLAN)
    if args.mode in ("train", "repair"):
        fit(doc, repair=args.mode == "repair")
    elif args.mode == "policy":
        calibrate(doc)
    elif args.mode == "coco":
        coco_transfer()
    else:
        evaluate(doc)
