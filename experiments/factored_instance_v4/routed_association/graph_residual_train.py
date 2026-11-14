"""Train matched MLPs on the frozen GraphLLM readouts."""
import argparse
from collections import Counter
import json

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from graph_residual_cache import FOLDER, PLAN, ROOT, SHARDS, digest, save_json
from learned_donor_train import SEED
from ownership_graph_train import optimize


ARMS = ("self", "residual", "max_entropy")
RECEIPT = FOLDER / "qualification.json"
PCA_RECEIPT = FOLDER / "qualification_pca.json"
PCA_PLAN = PLAN.with_name("GRAPH_RESIDUAL_PCA_ZH.md")


def raw(records):
    own = torch.cat([r["own"] for r in records]).float()
    residual = torch.cat([r["donor_residual"] for r in records]).float()
    supported = torch.cat([r["supported"] for r in records])
    assert own.shape == residual.shape and own.shape[:2] == supported.shape
    return torch.cat((own.flatten(1), residual.flatten(1), supported.float()), dim=1), own.shape[-1]


class AdmissionMLP(nn.Module):
    def __init__(self, hidden, use_residual, rank=0):
        super().__init__()
        self.hidden = hidden
        self.use_residual = use_residual
        self.rank = rank
        width = 4 * hidden + 2
        self.register_buffer("center", torch.zeros(width))
        self.register_buffer("scale", torch.ones(width))
        if rank:
            self.register_buffer("projection", torch.empty(4, hidden, rank))
            self.register_buffer("projected_scale", torch.ones(4))
        self.layers = nn.Sequential(nn.Linear(4 * rank + 2 if rank else width, 16), nn.GELU(), nn.Linear(16, 2))

    def normalize(self, x):
        values = (x - self.center) / self.scale
        flags = x[:, -2:]
        if self.rank:
            blocks = values[:, :-2].reshape(-1, 4, self.hidden)
            projected = torch.einsum("nbh,bhk->nbk", blocks, self.projection) / self.projected_scale[None, :, None]
            donor_flags = flags if self.use_residual else torch.zeros_like(flags)
            return torch.cat((projected[:, :2].flatten(1), (projected[:, 2:] * donor_flags[..., None]).flatten(1), donor_flags), dim=1)
        if not self.use_residual:
            return torch.cat((values[:, :2 * self.hidden], torch.zeros_like(values[:, 2 * self.hidden:])), dim=1)
        own = values[:, :2 * self.hidden]
        donor = values[:, 2 * self.hidden:-2].reshape(-1, 2, self.hidden)
        return torch.cat((own, (donor * flags[..., None]).flatten(1), flags), dim=1)

    def forward(self, x):
        return self.layers(self.normalize(x))


def load(splits):
    records, locks = [], {}
    for split in splits:
        for shard in range(SHARDS):
            path = FOLDER / f"{split}_{shard}.pt"
            item = torch.load(path, map_location="cpu", weights_only=True)
            assert item["lock"]["plan"] == digest(PLAN) and not item["eval_opened"] and item["split"] == split
            records.extend(item["records"])
            locks[path.name] = digest(path)
    x, hidden = raw(records)
    images = np.asarray([r["image"] for r in records])
    assert len(set(images)) == len(images)
    index = torch.repeat_interleave(torch.arange(len(images)), torch.tensor([len(r["actions"]) for r in records])).cuda()
    y = torch.tensor([z["target"] for r in records for z in r["outcomes"]], device="cuda")
    weights = {}
    for target, mass in ((1., .5), (0., .5), (.5, .5)):
        take = y == target
        count = torch.bincount(index[take], minlength=len(images))
        assert int((count > 0).sum()) > 0, (splits, target)
        w = y.new_zeros(len(y))
        w[take] = mass / int((count > 0).sum()) / count[index[take]].float()
        weights[target] = w
    assert len(index) == len(y) == len(x)
    return dict(x=x.cuda(), hidden=hidden, index=index, y=y, n=len(images), images=images,
                hard=weights[1.] + weights[0.], uncertainty=weights[.5], hashes=locks,
                counts=dict(Counter(str(float(v)) for v in y.tolist())))


@torch.no_grad()
def projection(x, hidden, rank):
    center = x.mean(0)
    basis, scale = [], []
    for block in range(4):
        span = slice(block * hidden, (block + 1) * hidden)
        data = x[:, span]
        if block >= 2:
            data = data[x[:, block - 4].bool()]
        center[span] = data.mean(0)
        data = data - center[span]
        _, _, vectors = torch.linalg.svd(data, full_matrices=False)
        top = vectors[:rank].T.contiguous()
        basis.append(top)
        scale.append((data @ top).square().mean().sqrt().clamp_min(.001))
    return center, torch.stack(basis), torch.stack(scale)


def fit(arm, train, cal, rank, pca):
    torch.manual_seed(SEED)
    model = AdmissionMLP(train["hidden"], arm != "self", rank).cuda()
    with torch.no_grad():
        if pca is not None:
            model.center.copy_(pca[0])
            model.projection.copy_(pca[1])
            model.projected_scale.copy_(pca[2])
        else:
            model.center.copy_(train["x"].mean(0))
            h = model.hidden
            for start in (0, h, 2 * h, 3 * h):
                block = train["x"][:, start:start + h] - model.center[start:start + h]
                model.scale[start:start + h] = block.square().mean().sqrt().clamp_min(.001)
        model.layers[-1].weight.zero_()
        model.layers[-1].bias.zero_()

    def loss(data):
        logits = model(data["x"])
        return F.binary_cross_entropy_with_logits(logits[:, 1] - logits[:, 0], data["y"], reduction="none")

    weight = train["hard"] + (train["uncertainty"] if arm == "max_entropy" else 0.)

    def objective():
        penalty = sum(layer.weight.square().sum() for layer in (model.layers[0], model.layers[-1]))
        return (loss(train) * weight).sum() + .05 * penalty

    @torch.no_grad()
    def measure():
        values = torch.zeros(cal["n"], device="cuda")
        values.scatter_add_(0, cal["index"], loss(cal) * cal["hard"] * cal["n"])
        assert torch.isfinite(values).all()
        return values.cpu().numpy()

    name = f"graph_residual_{arm}_pca" if rank else f"graph_residual_{arm}"
    result = optimize(name, model, objective, measure)
    result["checkpoint"] = digest(ROOT / f"{name}_best.pt")
    return result


def qualified(pca=False):
    path = PCA_RECEIPT if pca else RECEIPT
    doc = json.loads(path.read_bytes())
    assert doc["passed"] and not doc["eval_opened"] and doc["plan"] == digest(PLAN)
    if pca:
        assert doc["optimization_plan"] == digest(PCA_PLAN) and doc["rank"] == 16
    models = {}
    for arm in ARMS:
        name = f"graph_residual_{arm}_pca" if pca else f"graph_residual_{arm}"
        path = ROOT / f"{name}_best.pt"
        assert digest(path) == doc["arms"][arm]["checkpoint"]
        model = AdmissionMLP(doc["hidden"], arm != "self", doc.get("rank", 0))
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["model"])
        models[arm] = model.cuda().eval()
    return models, doc


def main(use_pca):
    path = PCA_RECEIPT if use_pca else RECEIPT
    assert not path.exists()
    train, cal = load(("train", "mine")), load(("cal",))
    assert train["hidden"] == cal["hidden"] and not np.intersect1d(train["images"], cal["images"]).size
    print("graph residual inputs", train["n"], train["counts"], cal["n"], cal["counts"], train["hidden"], flush=True)
    rank = 16 if use_pca else 0
    pca = projection(train["x"], train["hidden"], rank) if rank else None
    results = {arm: fit(arm, train, cal, rank, pca) for arm in ARMS}
    receipt = dict(passed=all(r["passed"] for r in results.values()), eval_opened=False, plan=digest(PLAN),
                   hidden=train["hidden"], training_images=train["n"], monitoring_images=cal["n"],
                   training=train["counts"], monitoring=cal["counts"], arms=results,
                   inputs={**train["hashes"], **cal["hashes"]})
    if rank:
        receipt.update(rank=rank, optimization_plan=digest(PCA_PLAN))
    save_json(path, receipt)
    print("all arm qualification", receipt["passed"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pca", action="store_true")
    main(parser.parse_args().pca)
