"""Learn an explicit unsupported-action state from real first choices."""
from collections import Counter
import hashlib
import json
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from receiver_frozen_train import NAME as PRIOR, ROOT, SHARDS, SEED, FrozenEvidence, FrozenReceiver, saved_native
from competition_cache import claim_targets
from counterfactual import ImageReplay, make_base
from dense_donor import vocabulary, save_json
from operational_cache import sources


NAME = "receiver_admission_v3"
PLAN = ROOT / "RECEIVER_ADMISSION_PLAN_ZH.md"
THRESHOLD = .9368263483047485
EPOCHS = 24
REJECT, FIX, HARM, OTHER = range(4)


class Admission(nn.Linear):
    def __init__(self):
        super().__init__(3, 4)
        nn.init.zeros_(self.weight)
        nn.init.zeros_(self.bias)

    def score(self, x):
        p = self(x).softmax(-1)
        return p[:, FIX] / (1 + 3 * p[:, HARM])


def qualified():
    qual = json.loads((ROOT / f"{NAME}_qualification.json").read_text())
    if not qual["passed"]:
        raise SystemExit("admission optimization qualification failed")
    path = ROOT / f"{NAME}_best.pt"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == qual["checkpoint"]
    assert hashlib.sha256(PLAN.read_bytes()).hexdigest() == qual["plan"] and not qual["eval_opened"]
    state = torch.load(path, map_location="cpu", weights_only=True)
    assert (state["epoch"], state["plan"], state["prior"]) == (qual["best_epoch"], qual["plan"], qual["prior"])
    head = Admission().eval()
    head.load_state_dict(state["model"])
    lock = {k: qual[k] for k in ("checkpoint", "plan", "prior")}
    return head, lock


def frozen():
    prior = json.loads((ROOT / f"{PRIOR}_qualification.json").read_text())
    path = ROOT / f"{PRIOR}_best.pt"
    assert prior["passed"] and not prior["eval_opened"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == prior["checkpoint"]
    state = torch.load(path, map_location="cpu", weights_only=True)
    assert state["epoch"] == prior["best_epoch"]
    model = FrozenReceiver().cuda().eval()
    model.load_state_dict(state["model"])
    return model, FrozenEvidence(state["text"]), state["names"], prior["checkpoint"]


def priors(model, geom, roles):
    geometry, state = model(geom, roles)
    align = geometry.softmax(1)[:, 1:].sum(1)
    neither, old, new, both = state.softmax(1).unbind(1)
    return torch.stack((align * new, align * (old + both), 1 - align + align * neither), dim=1).clamp_min(1e-6).log()


def first_index(q, c, u):
    return int(np.lexsort((c, q, -u))[0])


def eligible(value):
    return 0 < value < THRESHOLD


def single_features(model, reader, pack, base, q, c, slot):
    found = torch.nonzero(pack["ids"] == q).flatten()
    assert len(found) == 1 and int(base["query"][slot]) == q
    role = torch.tensor([[int(base["label"][slot]), c]], device="cuda")
    native = pack["logits"][q, role.cpu()].float().cuda()
    args = reader.extract(pack["grid"][found], pack["features"][pack["i"][found]], pack["boxes"][q:q + 1],
                          role, torch.zeros(1, dtype=torch.long, device="cuda"), native)
    value = priors(model, *args)
    assert value.shape == (1, 3) and torch.isfinite(value).all()
    return value[0].cpu()


def outcome(row, pack, base, manifest, names, q, c, slot):
    assert int(base["query"][slot]) == q and int(base["label"][slot]) != c
    replay = ImageReplay(row, pack, base, manifest, names)
    roles = torch.tensor([[int(base["label"][slot]), c]])
    support = claim_targets(row, manifest, names, replay.boxes[q:q + 1], roles)[0].tolist()
    action = ("replace", q, c, slot)
    new, lost, fp, sem, _ = replay.delta(action, float(replay.bs[slot]))
    unsafe = bool(sem < 0 or any(lost))
    state = HARM if unsafe else FIX if sem > 0 else REJECT if support == [0, 0] else OTHER
    return dict(label=state, correct=int(sem > 0), wrong=int(sem < 0), unsupported=int(support == [0, 0]),
                support=support, lost=[len(x) for x in lost], new=[len(x) for x in new], fp=fp)


@torch.inference_mode()
def materialize(splits, model, reader, expected_names):
    records = []
    for split in splits:
        source, tag, _ = sources(split)
        manifest = json.loads((source / "manifest.json").read_text())
        names, nfine, _ = vocabulary(manifest, tag)
        assert names[:nfine] == expected_names
        rows = {r["id"]: r for r in manifest["splits"][tag]}
        for shard in range(SHARDS):
            data = torch.load(ROOT / f"receiver_local_full_{split}_{shard}.pt", map_location="cpu", weights_only=True)
            keys = torch.column_stack((data["image"], data["query"], data["roles"], data["slot"])).numpy()
            native = saved_native(split, shard, keys)
            old = data["utility"].numpy()
            selected = []
            for image in np.unique(keys[:, 0]):
                ix = np.flatnonzero(keys[:, 0] == image)
                k = ix[first_index(keys[ix, 1], keys[ix, 3], old[ix])]
                if eligible(old[k]):
                    selected.append(k)
            ix = torch.tensor(selected, dtype=torch.long)
            rid, inverse = torch.unique(data["receiver"][ix], return_inverse=True)
            x = priors(model, *reader.extract(data["grid"][rid], data["own"][rid], data["box"][rid],
                                              data["roles"][ix].cuda(), inverse.cuda(), native[ix.cuda()])).cpu()
            for k, value in zip(selected, x):
                iid, q, old_cls, c, slot = map(int, keys[k])
                row = rows[iid]
                pack = torch.load(source / "cache" / tag / f"{iid}.pt", map_location="cpu", weights_only=True)
                base = make_base(pack, nfine)
                assert int(base["label"][slot]) == old_cls
                records.append(dict(image=iid, x=value.tolist(), **outcome(row, pack, base, manifest, names, q, c, slot)))
    assert len({r["image"] for r in records}) == len(records)
    return records


def main():
    path = ROOT / f"{NAME}_qualification.json"
    assert not path.exists()
    plan = hashlib.sha256(PLAN.read_bytes()).hexdigest()
    torch.manual_seed(SEED)
    model, reader, names, prior_sha = frozen()
    train, cal = materialize(("train", "mine"), model, reader, names), materialize(("cal",), model, reader, names)
    assert {r["image"] for r in train}.isdisjoint(r["image"] for r in cal)
    def arrays(records):
        return torch.tensor([r["x"] for r in records]), torch.tensor([r["label"] for r in records])
    x, y = arrays(train)
    cx, cy = arrays(cal)
    state = torch.bincount(y, minlength=4)
    assert (state > 0).all() and x.isfinite().all() and cx.isfinite().all()
    prior = (state / len(y)).log()
    losses = -prior[cy].numpy()
    group = np.random.default_rng(SEED).choice(losses, (2000, len(losses)), replace=True).mean(1)
    lo, hi = np.percentile(group, (2.5, 97.5))
    tol = max(.005, float(hi - lo) / 2)
    head = Admission()
    head.bias.data.copy_(prior)
    opt = torch.optim.LBFGS(head.parameters(), lr=1, max_iter=4, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-8, tolerance_change=1e-10)
    def objective():
        return F.cross_entropy(head(x), y) + .01 * head.weight.square().sum()
    def closure():
        opt.zero_grad(set_to_none=True)
        loss = objective()
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad.isfinite().all() for p in head.parameters())
        return loss
    def checkpoint(kind, epoch):
        saved = ROOT / f"{NAME}_{kind}.pt"
        torch.save(dict(model=head.state_dict(), epoch=epoch, plan=plan, prior=prior_sha), saved.with_suffix(".part"))
        saved.with_suffix(".part").replace(saved)
    best = math.inf
    history = []
    for epoch in range(1, EPOCHS + 1):
        opt.step(closure)
        with torch.no_grad():
            row = dict(epoch=epoch, train=float(objective()), cal=float(F.cross_entropy(head(cx), cy)))
        history.append(row)
        if row["cal"] < best:
            best, best_epoch = row["cal"], epoch
            checkpoint("best", epoch)
        print(row, flush=True)
    checkpoint("final", EPOCHS)
    save_json(ROOT / f"{NAME}_history.json", history)
    tr, dv = (np.array([r[k] for r in history[-5:]]) for k in ("train", "cal"))
    plateau = abs(tr[:2].mean() - tr[-2:].mean()) / tr[:2].mean()
    slope = abs(np.polyfit(np.arange(5), dv, 1)[0])
    checks = dict(schedule=len(history) == EPOCHS, finite=bool(np.isfinite(tr).all() and np.isfinite(dv).all()),
                  plateau=bool(plateau <= .025), slope=bool(slope <= tol / 4),
                  spread=bool(np.ptp(dv) <= tol), gap=bool(dv[-1] - best <= tol))
    result = dict(passed=all(checks.values()), checks=checks, best_cal=best, best_epoch=best_epoch,
                  final_cal=float(dv[-1]), baseline_cal=float(losses.mean()), tolerance=tol, plateau=float(plateau),
                  cal_slope=float(slope), plan=plan, prior=prior_sha,
                  checkpoint=hashlib.sha256((ROOT / f"{NAME}_best.pt").read_bytes()).hexdigest(),
                  train_states=dict(Counter(r["label"] for r in train)), cal_states=dict(Counter(r["label"] for r in cal)),
                  eval_opened=False)
    save_json(path, result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
