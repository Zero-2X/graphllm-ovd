"""Evaluate frozen donor-view scores on the registered Objects365 splits."""
from collections import Counter, defaultdict
from itertools import groupby
from pathlib import Path

import numpy as np
import torch
from torch.nn.functional import normalize

from objects365_cache import block_path, split_rows
from objects365_data import BLOCK, FOLDER, INDEX, SHARDS, digest, read, save_json
from objects365_donor_view_entropy import OUT, lock
from objects365_fit import replay_metrics
from omni_layer_train import Head
from ownership_graph_audit import auc

BUDGETS = (0, 1)
LEVEL = (0, 3)


def entropy(p):
    return -(p*p.clamp_min(1e-9).log() + (1-p)*(1-p).clamp_min(1e-9).log())


def collect(doc, split, anchor, model):
    first, outcomes, neighborhoods, original = [], [], [], []
    counts = Counter()
    for shard in range(SHARDS):
        frames = split_rows(doc, split, shard)
        for offset in range(0, len(frames), BLOCK):
            group = frames[offset:offset+BLOCK]
            part = offset//BLOCK
            source = block_path(split, shard, part)
            p = OUT/f"{split}_{shard}_{part:03d}.pt"
            block = torch.load(p, map_location="cpu", weights_only=True)
            assert block["lock"] == dict(lock=anchor, source=digest(source), images=[r["id"] for r in group])
            counts.update(block["counts"])
            cached = torch.load(source, map_location="cpu", weights_only=True)["records"]
            if split=="eval":
                original.extend(cached)
            for ref in cached:
                iid = ref["image"]
                if "actions" not in ref:
                    assert iid not in block["records"]
                    continue
                row = block["records"][iid]
                assert torch.equal(ref["actions"], row["actions"])
                views = row["views"].double()
                mean = views.mean(1)
                js = (entropy(mean)-entropy(views).mean(1)).clamp_min(0)
                multi = {i: verified for i, verified in zip(row["selected"].tolist(), row["donor_support"])}
                selected = int(mean.argmax())
                images_good = any(r["correct"] for r in ref["outcomes"])
                counts.update(possible=int(images_good), correct_first=ref["outcomes"][selected]["correct"],
                              multi_possible=int(any(ref["outcomes"][i]["correct"] for i in multi)),
                              multi_first=int(selected in multi), true_distinct_donor_pairs=sum(multi.values()))
                use, positions = [], []
                for j, (action, target) in enumerate(zip(row["actions"].tolist(), ref["outcomes"])):
                    harmful = bool(target["wrong"] or any(target["lost"]))
                    q, c, slot = action
                    entry = dict(target, image=iid, query=q, proposed=c, slot=slot, old=int(row["old"][j]),
                                 score=float(mean[j]), js=float(js[j]) if j in multi else None,
                                 harmful=harmful, donor_pair_verified=multi.get(j, False),
                                 quality=[float(row[k][j]) for k in ("receiver_iou", "score", "area")])
                    if j==selected:
                        first.append(entry)
                    if j in multi:
                        outcomes.append(entry)
                    if (entry["correct"] or harmful) and entry["quality"][0]>=.50:
                        use.append(entry)
                        positions.append(j)
                if use:
                    x = ref["x"][positions, 0].float()
                    both = normalize(x.reshape(-1, 4, 896), dim=2).reshape(-1, 3584)
                    last = both.reshape(-1, 2, 2, 896)[:, :, 0].reshape(-1, 1792)
                    hidden = normalize(model.net[:6](x), dim=1)
                    neighborhoods.extend(zip(use, normalize(last, dim=1), normalize(both, dim=1), hidden))
    assert counts["images"]==doc["counts"][split]
    assert len(first)==counts["candidate_images"] and len(outcomes)==counts["multiple"]
    return first, outcomes, neighborhoods, original, dict(counts)


def measure(rows, possible, eligible):
    total = len(rows)
    correct = sum(r["correct"] for r in rows)
    unknown = sum(r["target"]<0 for r in rows)
    harms = sum(r["harmful"] for r in rows)
    return dict(adopted=total, correct=correct, harmful=harms, unknown=unknown,
                unsupported=total-unknown-correct-harms, wrong=sum(r["wrong"] for r in rows),
                loss_images=sum(any(r["lost"]) for r in rows),
                new=[sum(r["new"][i] for r in rows) for i in LEVEL],
                lost=[sum(r["lost"][i] for r in rows) for i in LEVEL],
                fp=[sum(r["fp"][i] for r in rows) for i in LEVEL],
                confirmed_precision=correct/(total-unknown) if total>unknown else None,
                known_harm_rate=harms/(total-unknown) if total>unknown else None,
                coverage=total/eligible if eligible else 0,
                opportunity_recall=correct/possible if possible else None)


def prefix_curve(rows, cap, counts):
    base = [r for r in rows if r["score"]>.50 and (cap is None or r["js"] is None or r["js"]<=cap)]
    base.sort(key=lambda r:-r["score"])
    prefix, curve = [], [dict(cutoff=None, **measure([], counts["possible"], len(rows)))]
    for score, same in groupby(base, key=lambda r:r["score"]):
        prefix.extend(same)
        curve.append(dict(cutoff=score, **measure(prefix, counts["possible"], len(rows))))
    return curve


def best(curve, budget):
    return max((r for r in curve if r["harmful"]<=budget and r["loss_images"]==0),
               key=lambda r:(r["correct"], -r["unknown"], -r["unsupported"], -r["adopted"]))


def at_point(rows, cutoff, cap):
    if cutoff is None:
        return []
    return [r for r in rows if r["score"]>=cutoff and (cap is None or r["js"] is None or r["js"]<=cap)]


def ranking(rows):
    valid = [r for r in rows if r["correct"] or r["harmful"]]
    y = np.array([r["correct"] for r in valid])
    result = dict(valid=len(y), correct=int(y.sum()), harmful=int(len(y)-y.sum()))
    if 0<y.sum()<len(y):
        result.update(auc_mean=float(auc(np.array([r["score"] for r in valid]), y)),
                      auc_low_disagreement=float(auc(-np.array([r["js"] for r in valid]), y)))
    return result


def close(a, b, score=False):
    ai, ap, ar = a["quality"]
    bi, bp, br = b["quality"]
    return (abs(ai-bi)<=.15 and abs(ap-bp)<=.10 and max(ar,br)<=3*min(ar,br)
            and (not score or abs(a["score"]-b["score"])<=.05))


def conditional_js(rows):
    positives, harms = defaultdict(list), defaultdict(list)
    for r in rows:
        if r["quality"][0]>=.50 and (r["correct"] or r["harmful"]):
            (positives if r["correct"] else harms)[r["old"], r["proposed"]].append(r)
    used = set()
    counts = Counter()
    for key, group in harms.items():
        for bad in group:
            matches = [r for r in positives[key] if bad["image"] not in used and r["image"]!=bad["image"]
                       and r["image"] not in used and close(r,bad,score=True)]
            if matches:
                good = min(matches, key=lambda r:abs(r["score"]-bad["score"]))
                used.update((good["image"], bad["image"]))
                counts["pairs"] += 1
                counts["low_js_correct"] += good["js"]<bad["js"]
                counts["ties"] += good["js"]==bad["js"]
    return dict(counts)


def manifold(reference, tests):
    groups = defaultdict(list)
    for item in reference:
        r = item[0]
        groups[r["old"], r["proposed"]].append(item)
    scores = []
    for r, *representations in tests:
        candidates = [item for item in groups[r["old"], r["proposed"]] if close(r,item[0])]
        correct = {x[0]["image"] for x in candidates if x[0]["correct"]}
        harmful = {x[0]["image"] for x in candidates if x[0]["harmful"]}
        k = min(3, len(correct), len(harmful))
        if not k:
            continue
        distances = []
        for position, value in enumerate(representations,1):
            similarity = torch.stack([t[position] for t in candidates]) @ value
            by_outcome = {True:{}, False:{}}
            for ix in similarity.argsort(descending=True).tolist():
                other = candidates[ix][0]
                group = by_outcome[bool(other["correct"])]
                group.setdefault(other["image"], 1-float(similarity[ix]))
            distances.append(np.mean(list(by_outcome[False].values())[:k])
                             -np.mean(list(by_outcome[True].values())[:k]))
        scores.append((r["correct"], *distances))
    if not scores:
        return dict(covered=0, total=len(tests))
    y, *cols = np.array(scores).T
    result = dict(covered=len(y), total=len(tests), correct=int(y.sum()), harmful=int(len(y)-y.sum()))
    if 0<y.sum()<len(y):
        result["auc"] = {name:float(auc(column,y)) for name,column in zip(("last","concat","mlp"),cols)}
    return result


def paired_difference(doc, accepted):
    images = [r["id"] for r in doc["rows"] if r["split"]=="eval"]
    lookup = {image:i for i,image in enumerate(images)}
    rng = np.random.default_rng(20261112)
    samples = rng.integers(0,len(images),(2000,len(images)))
    results = {}
    for budget in BUDGETS:
        by_image = np.zeros((len(images), 4), dtype=int)
        for sign,name in ((-1,"mean"), (1,"mixed_entropy")):
            for row in accepted[name][budget]:
                by_image[lookup[row["image"]]] += sign*np.array(
                    [row["correct"], row["harmful"], row["target"]<0, row["new"][0]-row["lost"][0]],dtype=int)
        boot = by_image[samples].sum(1)
        names = ("correct", "harmful", "unknown", "net_iou50")
        results[budget] = {name:dict(delta=int(v.sum()), interval=np.quantile(b,(.025,.975)).tolist())
                           for name,v,b in zip(names,by_image.T,boot.T)}
    return results

@torch.inference_mode()
def main():
    doc = read(INDEX)
    repair = FOLDER/"qualification_repair.json"
    receipt = read(repair if repair.exists() else FOLDER/"qualification.json")
    checkpoint = FOLDER/receipt["model"]
    assert receipt["passed"] and receipt["index"]==digest(INDEX) and digest(checkpoint)==receipt["checkpoint"]
    anchor = lock(receipt["checkpoint"])
    model = Head().eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["model"])
    first, p_multi, reference, _, p_counts = collect(doc, "policy", anchor, model)
    positive_js = [r["js"] for r in first if r["score"]>.50 and r["correct"] and r["js"] is not None]
    cap = float(np.quantile(positive_js,.90)) if positive_js else None
    arms = {"mean":None, "mixed_entropy":cap}
    policy_curves = {name:prefix_curve(first,limit,p_counts) for name,limit in arms.items()}
    points = {name:{b:best(curve,b) for b in BUDGETS} for name,curve in policy_curves.items()}
    policy = dict(lock=anchor, entropy_cap=cap, calibration_correct_multi_first=len(positive_js),
                  counts=p_counts, points=points)
    save_json(OUT/"policy.json",policy)
    first, multi, tests, records, counts = collect(doc, "eval", anchor, model)
    curves = {name:prefix_curve(first,limit,counts) for name,limit in arms.items()}
    accepted = {name:{b:at_point(first,point["cutoff"],arms[name]) for b,point in rules.items()}
                for name,rules in points.items()}
    actual = {name:{b:measure(rows,counts["possible"],len(first)) for b,rows in budgets.items()}
              for name,budgets in accepted.items()}
    ap, classes = replay_metrics(doc,records,{name:choices[0] for name,choices in accepted.items()})
    result = dict(policy=policy, evaluator=digest(Path(__file__)), counts=counts, classes=classes,
                  actual=actual, ap=ap, paired=paired_difference(doc,accepted), accepted=accepted,
                  frontier={name:{b:best(curve,b) for b in BUDGETS} for name,curve in curves.items()},
                  curves=curves, multi_ranking=ranking(multi), policy_multi_ranking=ranking(p_multi),
                  true_donor_ranking=ranking([r for r in multi if r["donor_pair_verified"]]),
                  conditional_js=conditional_js(multi), manifold=manifold(reference,tests))
    save_json(OUT/"result.json",result)
    print("Development",counts,"metrics",actual,"conditional",result["conditional_js"],"manifold",result["manifold"],
          "AP",ap,flush=True)


if __name__=="__main__":
    torch.set_num_threads(4)
    main()


