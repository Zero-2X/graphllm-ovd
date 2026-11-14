"""Qualify witness reliability, identity edges and a three-state graph decision."""
import json

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from dense_donor import ROOT, SHARDS, save_json
from identity_train import pair_summary
from learned_donor_train import SEED
from ownership_graph_cache import EDGE_FIELDS, NODE_FIELDS
from ownership_witness import screen

EPOCHS = 24
RIDGE = .01


class Readout(nn.Module):
    def __init__(self, width, classes):
        super().__init__()
        self.linear = nn.Linear(width, classes)
        self.register_buffer("mean", torch.zeros(width))
        self.register_buffer("scale", torch.ones(width))
        self.register_buffer("adjust", torch.zeros(classes))

    def forward(self, x, corrected=False):
        z = self.linear(((x - self.mean) / self.scale).clamp(-6, 6))
        return z - self.adjust if corrected else z


def read(splits, index_receipt, prefix="ownership_graph_features"):
    parts, rows, nodes = [], 0, 0
    for split in splits:
        split_parts = []
        for shard in range(SHARDS):
            p = ROOT / f"{prefix}_{split}_{shard}.pt"
            d = torch.load(p, map_location="cpu", weights_only=True)
            assert d.pop("native_exact") and d.pop("node_fields") == NODE_FIELDS and d.pop("edge_fields") == EDGE_FIELDS
            d["group"] += rows
            d["links"] += nodes
            rows += len(d["rows"])
            nodes += len(d["node"])
            split_parts.append(d)
        y = torch.cat([d["rows"][:, -1] for d in split_parts]).long()
        info = index_receipt["splits"][split]
        assert torch.bincount(y, minlength=3).tolist() == info["group_targets"]
        assert sum(len(d["node"]) for d in split_parts) == info["witnesses"]
        parts.extend(split_parts)
    d = {k: torch.cat([p[k] for p in parts]).cuda() for k in parts[0] if k != "rows"}
    d["rows"] = torch.cat([p["rows"] for p in parts]).numpy()
    ids, image = np.unique(d["rows"][:, 0].astype(int), return_inverse=True)
    d["image"] = torch.from_numpy(image).cuda()
    d["ids"] = ids
    assert torch.equal(d["group"][d["links"][:, 0]], d["group"][d["links"][:, 1]])
    return d


def task(x, y, image, head, coefficient=1., active=None):
    n = int(image.max()) + 1
    if active is None:
        active = y >= 0
        counts = torch.bincount(image[active], minlength=n)
        scale = coefficient / max(1, int((counts > 0).sum()))
    else:
        counts = torch.bincount(image, minlength=n)
        scale = coefficient / n
    image = image[active]
    return dict(head=head, x=x[active], y=y[active].long(), image=image,
                weight=scale / counts[image].float())


def edge_tasks(data):
    image = data["image"][data["group"]]
    return [task(data["node"], data["node_y"], image, "node"),
            task(data["endpoint"].flatten(0, 1), data["endpoint_y"].flatten(), image.repeat_interleave(2), "edge", .5),
            task(data["edge"], data["edge_y"], image[data["links"][:, 0]], "edge", .5)]


@torch.no_grad()
def initialize(model, train):
    for name, readout in model.items():
        items = [d for d in train if d["head"] == name]
        x = torch.cat([d["x"] for d in items])
        y = torch.cat([d["y"] for d in items])
        w = torch.cat([d["weight"] for d in items])
        prior = torch.bincount(y, weights=w, minlength=len(readout.adjust))
        assert (prior > 0).all()
        prior /= prior.sum()
        balance = prior.rsqrt()
        balance /= (prior * balance).sum()
        readout.mean.copy_(x.mean(0))
        readout.scale.copy_(x.std(0).clamp_min(.001))
        readout.adjust.copy_(balance.log())
        readout.linear.weight.zero_()
        readout.linear.bias.copy_(prior.log() + readout.adjust)
        for d in items:
            d["training_weight"] = d["weight"] * balance[d["y"]]


@torch.inference_mode()
def monitor(model, tasks, n):
    loss = torch.zeros(n, device="cuda")
    for d in tasks:
        z = model[d["head"]](d["x"], corrected=True)
        value = F.cross_entropy(z, d["y"], reduction="none")
        assert torch.isfinite(z).all() and torch.isfinite(value).all()
        loss.scatter_add_(0, d["image"], value * d["weight"] * n)
    return loss.cpu().numpy()


def save(name, kind, model, epoch):
    path = ROOT / f"{name}_{kind}.pt"
    torch.save(dict(model=model.state_dict(), epoch=epoch), path.with_suffix(".part"))
    path.with_suffix(".part").replace(path)


def optimize(name, model, objective, measure):
    initial = measure()
    n = len(initial)
    rng = np.random.default_rng(SEED)
    bootstrap = rng.integers(0, n, size=(2000, n))
    lo, hi = np.percentile(initial[bootstrap].mean(1), [2.5, 97.5])
    tolerance = max(.005, float((hi - lo) / 2))
    opt = torch.optim.LBFGS(model.parameters(), lr=1., max_iter=4, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-7, tolerance_change=1e-10)

    def closure():
        opt.zero_grad(set_to_none=True)
        loss = objective()
        assert torch.isfinite(loss)
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters())
        return loss

    history, best = [], float("inf")
    for epoch in range(1, EPOCHS + 1):
        opt.step(closure)
        value = float(measure().mean())
        if value < best:
            best, best_epoch = value, epoch
            save(name, "best", model, epoch)
        with torch.no_grad():
            train_value = float(objective())
        history.append(dict(epoch=epoch, train=train_value, cal=value))
        save_json(ROOT / f"{name}_history.json", history)
        print(name, history[-1], flush=True)
    save(name, "final", model, EPOCHS)
    train_last = np.asarray([r["train"] for r in history[-5:]])
    cal_last = np.asarray([r["cal"] for r in history[-5:]])
    plateau = float(abs(train_last[:2].mean() - train_last[-2:].mean()) / train_last[:2].mean())
    slope = float(abs(np.polyfit(np.arange(5), cal_last, 1)[0]))
    checks = dict(schedule=len(history) == EPOCHS, finite=True, plateau=plateau <= .025,
                  cal_slope=slope <= tolerance / 4, cal_spread=float(np.ptp(cal_last)) <= tolerance,
                  cal_gap=float(cal_last[-1] - best) <= tolerance)
    result = dict(passed=all(checks.values()), checks=checks, initial_cal=float(initial.mean()),
                  final_cal=float(cal_last[-1]), best_cal=best, best_epoch=best_epoch,
                  cal_tolerance=tolerance, cal_slope=slope, plateau=plateau)
    save_json(ROOT / f"{name}_qualification.json", result)
    print(name, result, flush=True)
    if result["passed"]:
        payload = torch.load(ROOT / f"{name}_best.pt", map_location="cpu", weights_only=True)
        assert payload["epoch"] == best_epoch
        model.load_state_dict(payload["model"])
    return result


def qualify(name, model, train, cal, n):
    initialize(model, train)

    def objective():
        value = 0.
        for d in train:
            z = model[d["head"]](d["x"])
            assert torch.isfinite(z).all()
            value += (F.cross_entropy(z, d["y"], reduction="none") * d["training_weight"]).sum()
        return value + RIDGE / 2 * sum(h.linear.weight.square().sum() for h in model.values())

    return optimize(name, model, objective, lambda: monitor(model, cal, n))


@torch.inference_mode()
def hypothesis_values(model, data):
    group, links = data["group"], data["links"]
    reliability = model["node"](data["node"], corrected=True).softmax(-1)[:, 1]
    affiliation = model["edge"](data["endpoint"], corrected=True).softmax(-1)[:, :, 1]
    separate = model["edge"](data["edge"], corrected=True).softmax(-1)[:, 0]
    common = torch.minimum(reliability, affiliation.amin(1))
    preference = affiliation[:, 0] - affiliation[:, 1]
    left, right = links.T
    opposite = preference[left] * preference[right] < 0
    strongest = affiliation.amax(1)
    per_node = torch.minimum(reliability, strongest)
    distinct = torch.minimum(torch.minimum(per_node[left], per_node[right]), separate) * opposite
    return reliability, affiliation, separate, common, distinct


@torch.inference_mode()
def structure(model, data):
    group = data["group"]
    left = data["links"][:, 0]
    reliability, _, _, common, distinct = hypothesis_values(model, data)
    bank_size = torch.bincount(group, minlength=len(data["rows"]))
    x = reliability.new_zeros(len(bank_size), 2)
    x[:, 0].scatter_reduce_(0, group, common, reduce="amax", include_self=True)
    x[:, 1].scatter_reduce_(0, group[left], distinct, reduce="amax", include_self=True)
    active = bank_size > 0
    log = x.logit(eps=.0001)
    return torch.cat((log, log.amin(1, keepdim=True)), dim=1), active


def development(data, p, name="ownership_graph"):
    score = np.where(p[:, 1] > np.maximum(p[:, 0], p[:, 2]), p[:, 1], 0.)
    rows = data["rows"]
    index = {tuple(r[:14]): float(s) for r, s in zip(rows, score)}
    assert len(index) == len(rows)
    fixed = np.concatenate([np.load(ROOT / f"joint_affine_probe_{i}.npz")["rows"] for i in range(SHARDS)])
    cases = [dict(state=int(r[7]), box_iou=float(r[12]), score=index[tuple(r[:14])]) for r in fixed]
    matched = screen(cases, "score", True)
    order = np.argsort(-score, kind="stable")
    selected, best = np.zeros(len(score), dtype=bool), None
    for n, i in enumerate(order):
        if score[i] <= 0:
            break
        selected[i] = True
        if n + 1 < len(order) and score[order[n + 1]] == score[i]:
            continue
        effect = pair_summary(rows[:, :14], selected)
        if effect["different"]:
            break
        if effect["real_same"] >= 10 and effect["real_same"] / effect["real_accepted"] >= .90:
            if best is None or effect["real_same"] > best["real_same"]:
                best = dict(effect, threshold=float(score[i]))
    result = dict(pairs=len(rows), images=len(data["ids"]), matched=matched, threshold=best,
                  eligible=int((score > 0).sum()), passed=matched["passed"] and best is not None,
                  scope="old optimization development only")
    np.savez_compressed(ROOT / f"{name}_development.npz", rows=rows, probability=p, score=score)
    save_json(ROOT / f"{name}_result.json", result)
    print(result, flush=True)


def main():
    assert (ROOT / "OWNERSHIP_GRAPH_PLAN_ZH.md").is_file()
    assert not (ROOT / "ownership_graph_edges_qualification.json").exists()
    torch.manual_seed(SEED)
    receipt = json.loads((ROOT / "ownership_graph_index_receipt.json").read_text())
    train, cal = read(("train", "mine"), receipt), read(("cal",), receipt)
    assert not np.intersect1d(train["ids"], cal["ids"]).size
    n = len(cal["ids"])
    evidence = nn.ModuleDict(dict(node=Readout(len(NODE_FIELDS), 2), edge=Readout(len(EDGE_FIELDS), 2))).cuda()
    edge_status = qualify("ownership_graph_edges", evidence, edge_tasks(train), edge_tasks(cal), n)
    if not edge_status["passed"]:
        return
    ft, mt = structure(evidence, train)
    fc, mc = structure(evidence, cal)
    swapped, swapped_mask = structure(evidence, dict(cal, endpoint=cal["endpoint"].flip(1)))
    assert torch.equal(mc, swapped_mask) and torch.allclose(fc, swapped, atol=1e-6, rtol=1e-6)
    yt = torch.from_numpy(train["rows"][:, -1]).long().cuda()
    yc = torch.from_numpy(cal["rows"][:, -1]).long().cuda()
    assert (yt[~mt] == 0).all() and (yc[~mc] == 0).all()
    group = nn.ModuleDict(dict(group=Readout(ft.shape[1], 3))).cuda()
    group_status = qualify("ownership_graph_group", group, [task(ft, yt, train["image"], "group", active=mt)],
                           [task(fc, yc, cal["image"], "group", active=mc)], n)
    if group_status["passed"]:
        with torch.inference_mode():
            p = group["group"](fc, corrected=True).softmax(1).cpu().numpy()
        p[~mc.cpu().numpy()] = (1., 0., 0.)
        development(cal, p)


if __name__ == "__main__":
    main()
