"""Fit a larger MLP on the frozen cross-layer read token."""
import argparse
from collections import Counter
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from layer_entropy_cache import FOLDER, PLAN, SHARDS, digest, read
from route_screen import save_json


WIDTH, EPOCHS, BATCH, SEED = 3584, 40, 128, 20261111
RECEIPT = FOLDER / "qualification.json"
AVERAGE_RECEIPT = FOLDER / "qualification_average.json"
AVERAGE_PLAN = PLAN.with_name("LAYER_ENTROPY_AVERAGING_ZH.md")


class LayerMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(WIDTH)
        self.features = nn.Sequential(nn.Linear(WIDTH, 512), nn.GELU(), nn.Dropout(.2),
                                      nn.Linear(512, 128), nn.GELU())
        self.dropout = nn.Dropout(.2)
        self.output = nn.Linear(128, 2)

    def representation(self, x):
        return self.features(self.norm(x))

    def forward(self, x):
        return self.output(self.dropout(self.representation(x)))


def load(splits):
    records, hashes = [], {}
    counts = Counter()
    for split in splits:
        for shard in range(SHARDS):
            path = FOLDER / f"{split}_{shard}.pt"
            doc = torch.load(path, map_location="cpu", weights_only=True)
            assert doc["lock"]["plan"] == digest(PLAN) and doc["split"] == split and not doc["eval_opened"]
            counts.update(doc["counts"])
            records.extend(doc["records"])
            hashes[path.name] = digest(path)
    ids = np.array([r["image"] for r in records])
    assert len(set(ids)) == len(records)
    index = torch.repeat_interleave(torch.arange(len(records)), torch.tensor([len(r["y"]) for r in records])).cuda()
    y = torch.cat([r["y"] for r in records]).cuda()
    x = torch.cat([r["x"][:, 0] for r in records]).float().cuda()
    weight = torch.zeros(len(y), device="cuda")
    for label in (0, 1):
        take = y == label
        groups = torch.bincount(index[take], minlength=len(ids))
        weight[take] = .5 / int((groups > 0).sum()) / groups[index[take]].float()
    assert x.shape == (len(y), WIDTH) and len(y) == counts["positive"] + counts["negative"]
    assert abs(float(weight.sum()) - 1) < 1e-5
    return dict(x=x, y=y, weight=weight, index=index, ids=ids, counts=dict(counts), hashes=hashes)


@torch.inference_mode()
def image_losses(model, data):
    model.eval()
    losses = torch.zeros(len(data["ids"]), device="cuda")
    for start in range(0, len(data["y"]), 2048):
        s = slice(start, start+2048)
        values = F.cross_entropy(model(data["x"][s]), data["y"][s], reduction="none")
        losses.scatter_add_(0, data["index"][s], values * data["weight"][s] * len(losses))
    assert torch.isfinite(losses).all()
    return losses.cpu().numpy()


def checkpoint(model, epoch, kind):
    path = FOLDER / f"{kind}.pt"
    torch.save(dict(model=model.state_dict(), epoch=epoch, plan=digest(PLAN)), path.with_suffix(".part"))
    path.with_suffix(".part").replace(path)


def main(average=False):
    output = AVERAGE_RECEIPT if average else RECEIPT
    assert not output.exists()
    torch.manual_seed(SEED)
    train, cal = load(("train", "mine")), load(("cal",))
    assert not np.intersect1d(train["ids"], cal["ids"]).size
    print("training data", train["counts"], cal["counts"], flush=True)
    model = LayerMLP().cuda()
    model_avg = None
    epochs, suffix = EPOCHS, ""
    if average:
        original = read(RECEIPT)
        source = FOLDER / "best.pt"
        assert not original["passed"] and not original["eval_opened"] and original["checkpoint"] == digest(source)
        assert original["plan"] == digest(PLAN)
        model.load_state_dict(torch.load(source, map_location="cpu", weights_only=True)["model"])
        model_avg = torch.optim.swa_utils.AveragedModel(model)
        epochs, suffix = 16, "_average"
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5 if average else .001, weight_decay=.01)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    initial = image_losses(model, cal)
    n = len(initial)
    sample = np.random.default_rng(SEED).integers(0, n, size=(2000, n))
    lo, hi = np.percentile(initial[sample].mean(1), [2.5, 97.5])
    tol = max(.005, float((hi - lo) / 2))
    if average:
        tol = original["tolerance"]
    best, history = math.inf, []
    for epoch in range(1, epochs+1):
        model.train()
        order = torch.randperm(len(train["y"]), device="cuda")
        for indices in order.split(BATCH):
            opt.zero_grad(set_to_none=True)
            values = F.cross_entropy(model(train["x"][indices]), train["y"][indices], reduction="none")
            loss = (values * train["weight"][indices]).sum() * (len(order) / len(indices))
            assert torch.isfinite(loss)
            loss.backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters())
            opt.step()
        schedule.step()
        monitor = model
        if average:
            model_avg.update_parameters(model)
            monitor = model_avg.module
        cal_loss = float(image_losses(monitor, cal).mean())
        train_loss = float(image_losses(monitor, train).mean())
        if cal_loss < best:
            best, best_epoch = cal_loss, epoch
            checkpoint(monitor, epoch, "best"+suffix)
        history.append(dict(epoch=epoch, train=train_loss, cal=cal_loss, lr=schedule.get_last_lr()[0]))
        save_json(FOLDER / f"history{suffix}.json", history)
        print(history[-1], flush=True)
    checkpoint(monitor, epochs, "final"+suffix)
    train_tail = np.array([r["train"] for r in history[-5:]])
    cal_tail = np.array([r["cal"] for r in history[-5:]])
    plateau = float(abs(train_tail[:2].mean() - train_tail[-2:].mean()) / max(1e-8, train_tail[:2].mean()))
    slope = float(abs(np.polyfit(np.arange(5), cal_tail, 1)[0]))
    reference_best = min(best, original["best"]) if average else best
    checks = dict(schedule=len(history) == epochs and history[-1]["lr"] == 0., finite=True,
                  plateau=plateau <= .025, slope=slope <= tol/4, spread=float(np.ptp(cal_tail)) <= tol, gap=cal_loss-reference_best <= tol)
    receipt = dict(passed=all(checks.values()), checks=checks, plan=digest(PLAN), checkpoint=digest(FOLDER / f"best{suffix}.pt"),
                   train=train["counts"], monitoring=cal["counts"], inputs={**train["hashes"], **cal["hashes"]},
                   initial=float(initial.mean()), best=best, best_epoch=best_epoch, final=cal_loss, tolerance=tol,
                   plateau=plateau, slope=slope, parameters=sum(p.numel() for p in model.parameters()), eval_opened=False)
    if average:
        receipt.update(reference_best=reference_best, averaging_plan=digest(AVERAGE_PLAN))
    save_json(output, receipt)
    print("layer MLP qualification", {k:v for k,v in receipt.items() if k != "inputs"}, flush=True)


def qualified():
    receipt = AVERAGE_RECEIPT if AVERAGE_RECEIPT.exists() else RECEIPT
    info = read(receipt)
    path = FOLDER / ("best_average.pt" if receipt == AVERAGE_RECEIPT else "best.pt")
    if receipt == AVERAGE_RECEIPT:
        assert info["averaging_plan"] == digest(AVERAGE_PLAN)
    assert info["passed"] and not info["eval_opened"] and info["plan"] == digest(PLAN) and info["checkpoint"] == digest(path)
    saved = torch.load(path, map_location="cpu", weights_only=True)
    assert saved["plan"] == info["plan"] and saved["epoch"] == info["best_epoch"]
    model = LayerMLP().cuda().eval()
    model.load_state_dict(saved["model"])
    return model, dict(info, receipt=digest(receipt))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--average", action="store_true")
    main(parser.parse_args().average)
