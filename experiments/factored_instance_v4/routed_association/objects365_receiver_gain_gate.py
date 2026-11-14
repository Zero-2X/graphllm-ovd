"""Compare a verified-gain admission score with the frozen GraphLLM support score."""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn

from objects365_data import FOLDER, INDEX, digest, read, save_json
from objects365_donor_view_entropy_eval import measure, prefix_curve
from objects365_fit import qualified as graph_qualified, replay_metrics
from objects365_receiver_keep_gate import OUT as KEEP, SEED, EPOCHS, UNKNOWN_BUDGET
from objects365_receiver_keep_gate import evidence, features, harmful, losses, matched, measured
from objects365_receiver_keep_gate import objective, qualified as keep_qualified, read_rows, winners
from ownership_graph_audit import auc


OUT = FOLDER/"receiver_gain_gate"
PLAN = Path(__file__).with_name("OBJECTS365_RECEIVER_GAIN_GATE.md")


def anchor(graph):
    return dict(plan=digest(PLAN), code=digest(Path(__file__)), index=digest(INDEX),
                keep=digest(KEEP/"qualification.json"), keep_policy=digest(KEEP/"policy.json"),
                graph=graph["checkpoint"])


def gained(outcome):
    return bool(outcome["correct"] and not harmful(outcome) and (outcome["new"][0] or outcome["new"][3]))


class Gain(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("center", torch.zeros(17))
        self.register_buffer("scale", torch.ones(17))
        self.readout = nn.Linear(17,1)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def forward(self, x):
        return self.readout(((x-self.center)/self.scale).clamp(-6,6)).flatten()


@torch.inference_mode()
def rows_with_inputs(doc, splits, proposer):
    for split in splits:
        for _, records, claims, _, _ in read_rows(doc,split):
            for row in records:
                if "actions" not in row:
                    continue
                idx, raw = matched(row, claims.get(row["image"]))
                if not len(idx):
                    continue
                g = proposer(row["x"][idx,0].float().to(next(proposer.parameters()).device)).cpu()
                yield row, idx, torch.cat((features(raw),(g[:,1]-g[:,0])[:,None]),1)


def fit_data(doc, splits, proposer):
    xs, ys, ids, harms = [], [], [], []
    for row, idx, x in rows_with_inputs(doc,splits,proposer):
        outcomes = [row["outcomes"][j] for j in idx.tolist()]
        labeled = [o for o in outcomes if o["target"]>=0]
        if not labeled:
            continue
        xs.append(x[torch.tensor([o["target"]>=0 for o in outcomes])].clone())
        ys.extend(gained(o) for o in labeled)
        harms.extend(harmful(o) for o in labeled)
        ids.extend([row["image"]]*len(labeled))
    x,y = torch.cat(xs), torch.tensor(ys)
    image_ids, inverse = torch.unique(torch.tensor(ids),return_inverse=True)
    size = len(image_ids)
    weights = torch.zeros(len(y))
    for label in (False,True):
        use = y==label
        n = torch.bincount(inverse[use],minlength=size)
        assert use.any()
        weights[use] = .5/((n>0).sum()*n[inverse[use]])
    assert abs(float(weights.sum())-1)<1e-5
    counts = dict(images=size, actions=len(y), gains=int(y.sum()), harmful=sum(harms))
    return (x,y.float(),inverse,weights,size), counts, set(image_ids.tolist()), np.array(harms)


def save_weight(name, model, lock, step):
    path = OUT/f"{name}.pt"
    temporary = path.with_suffix(".part")
    torch.save(dict(model=model.state_dict(),anchor=lock,step=step),temporary)
    temporary.replace(path)
    return digest(path)


def fit(doc, graph, lock):
    train, tc, ti, _ = fit_data(doc,("train","rare_train"),graph)
    monitor, mc, mi, harm = fit_data(doc,("monitor",),graph)
    assert not ti&mi
    torch.manual_seed(SEED)
    model = Gain()
    model.center.copy_(train[0].mean(0))
    model.scale.copy_(train[0].std(0).clamp_min(.01))
    initial, values = measured(model,monitor)
    draws = np.random.default_rng(SEED).integers(0,len(values),(2000,len(values)))
    lo,hi = np.quantile(values[draws].mean(1),(.025,.975))
    tol = float(max(.005,(hi-lo)/2))
    optimizer = torch.optim.AdamW(model.parameters(),lr=.02,weight_decay=.05)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=EPOCHS)
    best, step, best_hash = initial, 0, save_weight("best",model,lock,0)
    history = []
    for iteration in range(1,EPOCHS+1):
        optimizer.zero_grad(set_to_none=True)
        loss = losses(model,train).mean()
        assert loss.isfinite()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        schedule.step()
        tr,_ = measured(model,train)
        valid,_ = measured(model,monitor)
        history.append(dict(step=iteration,train=tr,monitor=valid,lr=schedule.get_last_lr()[0]))
        if valid<best:
            best,step,best_hash = valid,iteration,save_weight("best",model,lock,iteration)
    final_hash = save_weight("final",model,lock,EPOCHS)
    last = np.array([v["monitor"] for v in history[-5:]])
    losses_train = [v["train"] for v in history[-5:]]
    checks = dict(schedule=history[-1]["lr"]==0,
                  plateau=abs(losses_train[-1]-losses_train[0])/losses_train[0]<=.025,
                  slope=abs(np.polyfit(np.arange(5),last,1)[0])<=tol/4,
                  spread=float(np.ptp(last))<=tol, gap=last[-1]-best<=tol)
    model.load_state_dict(torch.load(OUT/"best.pt",map_location="cpu",weights_only=True)["model"])
    with torch.no_grad():
        score = model(monitor[0]).numpy()
    positive = monitor[1].numpy()>0
    use = positive|harm
    receipt = dict(anchor=lock,passed=bool(all(checks.values()) and step>0),
                   checks={k:bool(v) for k,v in checks.items()},train=tc,monitor=mc,
                   initial=initial,best=best,best_step=step,best_hash=best_hash,
                   final_hash=final_hash,final=history[-1],tolerance=tol,
                   monitor_auc=auc(score,positive),monitor_gain_vs_harm_auc=auc(score[use],positive[use]))
    save_json(OUT/"history.json",history)
    save_json(OUT/"qualification.json",receipt)
    print(receipt,flush=True)


def load_model(lock):
    receipt = read(OUT/"qualification.json")
    assert receipt["passed"] and receipt["anchor"]==lock
    assert receipt["best_hash"]==digest(OUT/"best.pt")
    state = torch.load(OUT/"best.pt",map_location="cpu",weights_only=True)
    assert state["anchor"]==lock and state["step"]==receipt["best_step"]
    model = Gain().eval()
    model.load_state_dict(state["model"])
    return model


@torch.inference_mode()
def candidates(doc, split, model, graph):
    keep, _ = keep_qualified()
    data, native, counts = evidence(doc,split,keep)
    cfg = read(KEEP/"policy.json")["points"]["mixed_entropy"]["limited"]
    baseline = winners(data,cfg["risk_cap"],cfg["js_cap"])
    baseline = [r for r in baseline if r["score"]>=cfg["point"]["cutoff"]]
    for item in data.values():
        item["mean"] = torch.full_like(item["mean"],-torch.inf)
    matched_actions = 0
    for row, idx, x in rows_with_inputs(doc,(split,),graph):
        item = data[row["image"]]
        assert torch.equal(item["query"][idx],row["actions"][idx,0])
        assert torch.equal(item["proposal"][idx],row["actions"][idx,1])
        item["mean"][idx] = model(x).sigmoid().to(item["mean"].dtype)
        matched_actions += len(idx)
    assert matched_actions==counts["matched"]
    return winners(data,cfg["risk_cap"],cfg["js_cap"]), baseline, native, counts


def policy(doc, model, graph):
    choices, baseline, _, counts = candidates(doc,"policy",model,graph)
    # The utility probability is a class-balanced ranking score, not a 0.5 gate.
    # Prefixes are monotone transforms of the ranking score so every eligible prefix is considered.
    transformed = [dict(r,score=.5+.5*r["score"]) for r in choices]
    curve = prefix_curve(transformed,None,counts)
    points = {}
    for budget,unknown in UNKNOWN_BUDGET.items():
        eligible = [r for r in curve if not (r["harmful"] or r["loss_images"]) and r["unknown"]<=unknown]
        point = max(eligible,key=objective)
        points[budget] = point
    result = dict(qualification=digest(OUT/"qualification.json"),points=points,counts=counts,
                  baseline=measure(baseline,counts["possible"],counts["candidate_images"]))
    save_json(OUT/"policy.json",result)
    print(result,flush=True)


def evaluate(doc, model, graph):
    settings = read(OUT/"policy.json")
    assert settings["qualification"]==digest(OUT/"qualification.json")
    rows,baseline,native,counts = candidates(doc,"eval",model,graph)
    selected = dict(baseline=baseline)
    for name,p in settings["points"].items():
        t = p["cutoff"]
        selected[name] = [] if t is None else [r for r in rows if .5+.5*r["score"]>=t]
    outcomes = {k:measure(v,counts["possible"],counts["candidate_images"]) for k,v in selected.items()}
    ap,classes = replay_metrics(doc,native,selected)
    result = dict(policy=digest(OUT/"policy.json"),counts=counts,outcomes=outcomes,ap=ap,
                  classes=classes,selected=selected)
    save_json(OUT/"result.json",result)
    print(outcomes,ap,flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage",choices=("fit","policy","eval"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    OUT.mkdir(exist_ok=True)
    doc = read(INDEX)
    graph, receipt = graph_qualified()
    lock = anchor(receipt)
    if args.stage=="fit":
        fit(doc,graph,lock)
    else:
        (policy if args.stage=="policy" else evaluate)(doc,load_model(lock),graph)
