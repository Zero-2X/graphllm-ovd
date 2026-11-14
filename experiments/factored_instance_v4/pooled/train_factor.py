"""Fit the three factorized readouts on the locked v1 development split."""
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from factor import FactoredGate, FactorEvidence, V3
from counterfactual import make_base, save_json, vocabulary
from paired_targets import V1

SEED = 20261110
EPOCHS = 24
BATCH = 1024
LR = .001
TASKS = ("alignment", "semantic", "effect")


def by_image(rows):
    ids, starts, counts = np.unique(rows[:, 0], return_index=True, return_counts=True)
    return {int(k): rows[s:s+n] for k, s, n in zip(ids, starts, counts)}


def extract(manifest, split, reader):
    path = Path("factors_"+split+".pt")
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=True)
    with np.load(V3/("receiver_targets_"+split+".npz")) as data:
        receivers = by_image(data["receiver"])
        claims = by_image(data["claims"])
    with np.load(V3/("paired_targets_"+split+".npz")) as data:
        c = {str(k): i for i, k in enumerate(data["columns"])}
        effects = by_image(data["rows"])
    names, nfine, parents = vocabulary(manifest, split)
    reader.vocabulary(names, nfine, parents)
    buffers = {k: [] for k in ("grid", "own", "x", "same", "q", "s", "a", "meta", "raw_conf")}
    r_offset = c_offset = 0
    for i, row in enumerate(manifest["splits"][split], 1):
        iid = row["id"]
        recv = receivers.get(iid, ())
        cl = claims.get(iid, ())
        eff = effects.get(iid, ())
        if not (len(recv) or len(cl) or len(eff)):
            continue
        pack = torch.load(V1/"cache"/split/(str(iid)+".pt"), map_location="cpu", weights_only=True)
        base = make_base(pack, nfine)
        r = [int(t[2]) for t in recv]
        edges = [(int(t[2]), int(t[3])) for t in cl]
        actions, enc = reader.encode(pack, base, edges, r)
        buffers["grid"].append(enc["grid"])
        buffers["own"].append(enc["own"])
        buffers["x"].append(enc["x"])
        buffers["same"].append(enc["same"])
        old = {int(t[2]): int(t[7]) for t in recv}
        loc = [(enc["r_index"][j]+r_offset, int(t[3]), iid) for j, t in zip(r, recv)]
        sem = [(enc["c_index"][edge]+c_offset, old[edge[0]]+2*int(t[5]), int(t[6]), iid)
               for edge, t in zip(edges, cl)]
        buffers["q"].append(torch.tensor(loc, dtype=torch.long).reshape(-1, 3))
        buffers["s"].append(torch.tensor(sem, dtype=torch.long).reshape(-1, 4))
        buffers["raw_conf"].append(pack["logits"][pack["ids"][r], :nfine].float().amax(-1).sigmoid())
        lookup = {(q, category, round(1000*score)): j for j, ((_, q, category, _), score) in enumerate(actions)}
        indices, targets = [], []
        for t in eff:
            j = lookup[int(t[c["query"]]), int(t[c["class"]]), int(t[c["score_milli"]])]
            slot = actions[j][0][3]
            assert t[c["native_slot"]] == (-1 if slot is None else slot)
            harm = int(t[c["harm"]])
            intent = int(t[c["receiver_proposal_correct"]])
            safe = int(t[c["gain"]]) and not harm
            state = 2 if harm else int(safe)
            targets.append((int(enc["rec"][j])+r_offset, int(enc["claim"][j])+c_offset, intent, state, iid))
            indices.append(j)
        buffers["a"].append(torch.tensor(targets, dtype=torch.long).reshape(-1, 5))
        buffers["meta"].append(enc["meta"][indices])
        r_offset += len(enc["grid"])
        c_offset += len(enc["x"])
        if i % 160 == 0 or i == len(manifest["splits"][split]):
            print("features", split, i, flush=True)
    result = {k: torch.cat(v) for k, v in buffers.items()}
    assert len(result["q"]) == sum(map(len, receivers.values()))
    assert len(result["s"]) == sum(map(len, claims.values()))
    assert len(result["a"]) == sum(map(len, effects.values())) == len(result["meta"])
    aligned = result["s"]
    assert torch.equal(result["same"][aligned[:, 0]] & (aligned[:, 1]>0), aligned[:, 1]==3)
    result["context_dim"] = reader.width
    temp = path.with_suffix(".part")
    torch.save(result, temp)
    temp.replace(path)
    return result


def loss_quality(model, data, i):
    r, y, _ = data["q"][i].unbind(-1)
    return F.cross_entropy(model.alignment_logits(data["grid"][r], data["own"][r]), y, reduction="none")


def loss_semantic(model, data, i):
    claim, y, donor, _ = data["s"][i].unbind(-1)
    z, d = model.semantic_logits(data["x"][claim], data["same"][claim])
    return F.cross_entropy(z, y, reduction="none") + .25*F.binary_cross_entropy_with_logits(d, donor.float(), reduction="none")


def loss_effect(model, data, i):
    r, c, intent, state, _ = data["a"][i].unbind(-1)
    with torch.no_grad():
        q = model.alignment_logits(data["grid"][r], data["own"][r]).softmax(-1)
        s, _ = model.semantic_logits(data["x"][c], data["same"][c])
    z = model.effect_logits(q, s.softmax(-1), data["meta"][i])
    valid = F.cross_entropy(z[:, :3], state, reduction="none")
    invalid = F.binary_cross_entropy_with_logits(z[:, 3], (state == 2).float(), reduction="none")
    return torch.where(intent.bool(), valid, invalid)


LOSSES = dict(zip(("q", "s", "a"), (loss_quality, loss_semantic, loss_effect)))


@torch.inference_mode()
def validate(model, data):
    model.eval()
    values = []
    for key, fn in LOSSES.items():
        result = torch.cat([fn(model, data, i).cpu() for i in torch.arange(len(data[key]), device="cuda").split(BATCH)])
        assert torch.isfinite(result).all()
        values.append(result.numpy())
    return values


def tolerance(group, diff):
    _, inv = np.unique(group, return_inverse=True)
    counts = np.bincount(inv)
    total = np.bincount(inv, weights=diff)
    n = len(counts)
    ids = np.random.default_rng(SEED).integers(n, size=(1000, n))
    se = (total[ids].sum(1)/counts[ids].sum(1)).std(ddof=1)
    return float(max(.005, 2*se))


def checkpoint(path, model, dim, context, epoch):
    tmp = Path(path+".part")
    torch.save(dict(model=model.state_dict(), dim=dim, context_dim=context, epoch=epoch), tmp)
    tmp.replace(path)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_float32_matmul_precision("high")
    manifest = json.loads((V1/"manifest.json").read_text())
    reader = None if all(Path("factors_"+s+".pt").is_file() for s in ("train", "cal")) else FactorEvidence()
    train = extract(manifest, "train", reader)
    cal = extract(manifest, "cal", reader)
    del reader
    torch.cuda.empty_cache()
    dim, context = train["x"].shape[1], train["context_dim"]
    assert cal["x"].shape[1] == dim
    groups = [cal[k][:, -1].numpy() for k in LOSSES]
    train = {k: v.cuda() for k, v in train.items() if torch.is_tensor(v)}
    cal = {k: v.cuda() for k, v in cal.items() if torch.is_tensor(v)}
    steps = (max(len(train[k]) for k in LOSSES)+BATCH-1)//BATCH
    assert steps <= min(len(train[k]) for k in LOSSES)
    model = FactoredGate(dim, context).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=LR*.025)
    before = validate(model, cal)
    best = float("inf")
    history = []
    for epoch in range(1, EPOCHS+1):
        model.train()
        accum = torch.zeros(3, device="cuda")
        batches = [torch.randperm(len(train[k]), device="cuda").tensor_split(steps) for k in LOSSES]
        for ids in zip(*batches):
            opt.zero_grad(set_to_none=True)
            losses = torch.stack([fn(model, train, i).mean() for fn, i in zip(LOSSES.values(), ids)])
            loss = losses.mean()
            assert torch.isfinite(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            opt.step()
            accum += losses.detach()
        scheduler.step()
        after = validate(model, cal)
        per_cal = [float(x.mean()) for x in after]
        value = float(np.mean(per_cal))
        item = dict(epoch=epoch, train=float(accum.mean())/steps, cal=value,
                    cal_components=per_cal, lr=opt.param_groups[0]["lr"], train_components=(accum/steps).tolist())
        history.append(item)
        if value < best:
            best = value
            checkpoint("best.pt", model, dim, context, epoch)
        save_json("history.json", history)
        print(item, flush=True)
    checkpoint("final.pt", model, dim, context, EPOCHS)
    t = np.array([x["train"] for x in history[-5:]])
    plateau = float(abs(t[:2].mean()-t[-2:].mean())/t[:2].mean())
    components = {}
    for j, name in enumerate(TASKS):
        scores = np.array([x["cal_components"][j] for x in history])
        tail = scores[-5:]
        tol = tolerance(groups[j], after[j]-before[j])
        slope = float(abs(np.polyfit(np.arange(5), tail, 1)[0]))
        spread = float(np.ptp(tail))
        gap = float(scores[-1]-scores.min())
        components[name] = dict(initial=float(before[j].mean()), best=float(scores.min()), final=float(scores[-1]),
                                slope=slope, spread=spread, gap=gap, tolerance=tol,
                                passed=slope<=tol/4 and spread<=tol and gap<=tol)
    checks = dict(schedule=abs(history[-1]["lr"]-LR*.025)<1e-9, finite=True, plateau=plateau<=.025,
                  component_stability=all(x["passed"] for x in components.values()))
    result = dict(passed=all(checks.values()), checks=checks, components=components, best_cal=best,
                  final_cal=history[-1]["cal"], plateau=plateau,
                  train_counts={task:len(train[key]) for task,key in zip(TASKS,LOSSES)},
                  cal_counts={task:len(cal[key]) for task,key in zip(TASKS,LOSSES)}, epochs=EPOCHS)
    save_json("qualification.json", result)
    print("qualification", result, flush=True)


if __name__ == "__main__":
    main()
