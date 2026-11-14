"""Fit a small shared role readout over frozen receiver evidence."""
import hashlib
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from receiver_local_train import ROOT, SHARDS, V1, SEED, bootstrap, class_text, likelihood, load, pairs, save_json, utility
from relabel import read_factor
from semantic_graph_refresh import PREFIX


NAME = "receiver_frozen_v2"
PLAN = ROOT / "RECEIVER_FROZEN_TRAIN_PLAN_ZH.md"
EPOCHS = 24


class FrozenReceiver(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.zeros(()))
        self.geometry_bias = nn.Parameter(torch.zeros(2))
        self.role = nn.Linear(3, 1)
        nn.init.zeros_(self.role.weight)
        nn.init.zeros_(self.role.bias)
        self.both = nn.Parameter(torch.zeros(()))

    def forward(self, geometry, evidence):
        bias = F.pad(self.geometry_bias, (1, 0))
        geometry = (1 + .5 * self.temperature.tanh()) * geometry + bias
        old, new = self.role(evidence).squeeze(-1).unbind(1)
        states = torch.stack((torch.zeros_like(old), old, new, old + new + self.both), dim=1)
        return geometry, states


class FrozenEvidence:
    def __init__(self, text):
        self.text = text.cuda()
        self.alignment = read_factor(ROOT.parent / "pooled" / "best.pt").eval()

    @torch.no_grad()
    def extract(self, grid, own, box, roles, receiver, native):
        align = []
        for g, x, b in zip(grid.split(2048), own.split(2048), box.split(2048)):
            g, x, b = g.cuda(), x.cuda().float(), b.cuda()
            wh = (b[:, 2:] - b[:, :2]).clamp_min(1e-6)
            prior_own = torch.cat((x[:, :256], b, wh, wh.prod(1, keepdim=True)), dim=1).half()
            align.append(self.alignment.alignment_logits(g, prior_own))
        geometry = torch.cat(align)
        clip = F.normalize(own.cuda()[:, 256:].float(), dim=1) @ self.text.T
        features = []
        classes = torch.arange(len(self.text), device="cuda")
        for ix in torch.arange(len(roles), device="cuda").split(2048):
            r, c = receiver[ix], roles[ix]
            fit = clip[r]
            absolute = fit.gather(1, c)
            other = fit[:, None].masked_fill(c[..., None] == classes, -torch.inf).amax(2)
            features.append(torch.stack((native[ix] / 3, 8 * absolute, 16 * (absolute - other)), dim=2))
        return geometry[receiver], torch.cat(features)


def saved_native(split, shard, expected):
    with np.load(ROOT / f"{PREFIX}_{split}_{shard}.npz") as raw:
        fields = list(raw["fields"])
        rows = raw["rows"]
        x = raw["evidence"]
        assert np.array_equal(rows[:, [0, 1, 2, 3, 5]], expected)
        old = x[:, fields.index("old_native")]
        new = old + x[:, fields.index("delta_native")]
    return torch.from_numpy(np.column_stack((old, new))).cuda()


def prepare(splits, reader):
    data = load(splits)
    keys = torch.column_stack((data["image"], data["query"], data["roles"], data["slot"])).cpu().numpy()
    native = []
    offset = 0
    for split in splits:
        for shard in range(SHARDS):
            with np.load(ROOT / f"{PREFIX}_{split}_{shard}.npz") as src:
                count = len(src["rows"])
            native.append(saved_native(split, shard, keys[offset:offset + count]))
            offset += count
    assert offset == len(keys)
    data["fixed"] = reader.extract(*(data.pop(k) for k in ("grid", "own", "box", "roles", "receiver")), torch.cat(native))
    assert all(torch.isfinite(v).all() for v in data["fixed"])
    return data


def nll(logits, data):
    terms = likelihood(*logits, data, slice(None))
    return tuple(t.sum() / len(data["images"]) for t in terms)


@torch.no_grad()
def measure(model, data):
    geometry, role = nll(model(*data["fixed"]), data)
    return dict(total=float(geometry + role), geometry=float(geometry), role=float(role))


def checkpoint(model, names, text, kind, epoch, plan):
    path = ROOT / f"{NAME}_{kind}.pt"
    torch.save(dict(model=model.state_dict(), names=names, text=text, epoch=epoch, plan=plan), path.with_suffix(".part"))
    path.with_suffix(".part").replace(path)


def main():
    plan = hashlib.sha256(PLAN.read_bytes()).hexdigest()
    path = ROOT / f"{NAME}_qualification.json"
    assert not path.exists()
    torch.manual_seed(SEED)
    names, text = class_text()
    reader = FrozenEvidence(text)
    train, cal = prepare(("train", "mine"), reader), prepare(("cal",), reader)
    assert len(train["image"]) == 122079 and len(cal["image"]) == 14325
    assert set(train["images"].tolist()).isdisjoint(cal["images"].tolist())
    matched, same = pairs(train)
    baseline, tol = bootstrap(train, cal)
    model = FrozenReceiver().cuda()
    initial = measure(model, cal)
    opt = torch.optim.LBFGS(model.parameters(), lr=1., max_iter=4, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-7, tolerance_change=1e-10)

    def loss():
        logits = model(*train["fixed"])
        fit = sum(nll(logits, train))
        u = utility(*logits)[matched]
        rank = F.softplus(2 * (u[:, 1] - u[:, 0])).mean()
        return fit + .04 * rank + .005 * sum(p.square().sum() for p in model.parameters())

    def closure():
        opt.zero_grad(set_to_none=True)
        value = loss()
        assert torch.isfinite(value)
        value.backward()
        assert all(p.grad.isfinite().all() for p in model.parameters())
        return value

    history, best = [], math.inf
    for epoch in range(1, EPOCHS + 1):
        opt.step(closure)
        valid = measure(model, cal)
        with torch.no_grad():
            row = dict(epoch=epoch, train=float(loss()), cal=valid)
        history.append(row)
        if valid["total"] < best:
            best, best_epoch = valid["total"], epoch
            checkpoint(model, names, text, "best", epoch, plan)
        print(row, flush=True)
    checkpoint(model, names, text, "final", EPOCHS, plan)
    save_json(ROOT / f"{NAME}_history.json", history)
    tr = np.array([x["train"] for x in history[-5:]])
    dv = np.array([x["cal"]["total"] for x in history[-5:]])
    plateau = abs(tr[:2].mean() - tr[-2:].mean()) / tr[:2].mean()
    slope = abs(np.polyfit(np.arange(5), dv, 1)[0])
    checks = dict(schedule=len(history) == EPOCHS, finite=all(math.isfinite(x) for x in (*tr, *dv)),
                  plateau=bool(plateau <= .025), slope=bool(slope <= tol / 4),
                  spread=bool(np.ptp(dv) <= tol), gap=bool(dv[-1] - best <= tol))
    result = dict(passed=all(checks.values()), checks=checks, plan=plan, initial_cal=initial, baseline_cal=baseline,
                  best_cal=best, best_epoch=best_epoch, final_cal=history[-1]["cal"], cal_tolerance=tol,
                  cal_slope=float(slope), plateau=float(plateau), paired=len(matched), same_image_pairs=same,
                  checkpoint=hashlib.sha256((ROOT / f"{NAME}_best.pt").read_bytes()).hexdigest(), eval_opened=False)
    save_json(path, result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
