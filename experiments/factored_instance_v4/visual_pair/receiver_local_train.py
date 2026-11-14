"""Learn geometry and competing class support from the same receiver pixels."""
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoProcessor, CLIPModel

from dense_donor import ROOT, SHARDS, V1, save_json, vocabulary


NAME = "receiver_local_v1"
PLAN = ROOT / "RECEIVER_LOCAL_TRAIN_PLAN_ZH.md"
EPOCHS, BATCH, SEED = 40, 512, 41


class ReceiverLocal(nn.Module):
    def __init__(self, text):
        super().__init__()
        self.register_buffer("text", text.clone())
        axis = torch.linspace(-1, 1, 5)
        self.register_buffer("position", torch.stack(torch.meshgrid(axis, axis, indexing="ij"), dim=-1).reshape(25, 2))
        width = 64
        self.grid_norm = nn.LayerNorm(256)
        self.patch = nn.Linear(258, width)
        self.query = nn.Linear(256, width)
        self.class_text = nn.Linear(512, width, bias=False)
        self.geometry = nn.Sequential(nn.Linear(width * 2 + 2, 48), nn.SiLU(), nn.Linear(48, 3))
        self.role = nn.Sequential(nn.Linear(width * 3 + 1, 48), nn.SiLU(), nn.Dropout(.1), nn.Linear(48, 1))
        self.both = nn.Parameter(torch.zeros(()))

    def forward(self, grid, own, box, roles):
        grid = self.grid_norm(grid.float())
        position = self.position[None].expand(len(grid), -1, -1)
        local = F.silu(self.patch(torch.cat((grid, position), dim=-1)))
        own = own.float()
        query = F.silu(self.query(own[:, :256]))
        directions = F.normalize(local, dim=-1)
        align = torch.einsum("bnd,bd->bn", directions, F.normalize(query, dim=-1)) * 4
        region = torch.einsum("bn,bnd->bd", align.softmax(1), local)
        size = (box[:, 2:] - box[:, :2]).clamp_min(1e-6).log()
        geometry = self.geometry(torch.cat((query, region, size), dim=1))

        text = self.text[roles]
        role_query = F.normalize(self.class_text(text), dim=-1)
        match = torch.einsum("bnd,bkd->bkn", directions, role_query) * 4
        weights = (match + align[:, None]).softmax(dim=-1)
        patch = torch.einsum("bkn,bnd->bkd", weights, local)
        clip = torch.einsum("bd,bkd->bk", F.normalize(own[:, 256:], dim=-1), text)
        evidence = torch.cat((patch, role_query, patch * role_query, (4 * clip)[..., None]), dim=-1)
        old, new = self.role(evidence).squeeze(-1).unbind(1)
        state = torch.stack((torch.zeros_like(old), old, new, old + new + self.both), dim=1)
        return geometry, state


def utility(geometry, state):
    p_align = geometry.softmax(1)[:, 1:].sum(1)
    _, old, new, both = state.softmax(1).unbind(1)
    return p_align * (new - old - both)


@torch.inference_mode()
def class_text():
    manifest = json.loads((V1 / "manifest.json").read_text())
    names, nfine, _ = vocabulary(manifest, "cal")
    assert nfine == 48
    names = names[:nfine]
    path = Path(os.environ["BASE"]) / "models" / "clip"
    processor = AutoProcessor.from_pretrained(path, local_files_only=True)
    clip = CLIPModel.from_pretrained(path, local_files_only=True, torch_dtype=torch.float16,
                                     use_safetensors=True).cuda().eval()
    prompt = processor(text=[f"a photo of a {name}." for name in names], padding=True,
                       return_tensors="pt").to("cuda")
    return names, F.normalize(clip.get_text_features(**prompt).float(), dim=-1).cpu()


def load(splits):
    packs, offset = [], 0
    for split in splits:
        for shard in range(SHARDS):
            path = ROOT / f"receiver_local_full_{split}_{shard}.pt"
            data = torch.load(path, map_location="cpu", weights_only=True)
            data["receiver"] += offset
            offset += len(data["own"])
            packs.append(data)
    fields = ("grid", "own", "box", "receiver", "roles", "image", "query", "slot", "alignment", "supports",
              "hard_fix", "hard_bad", "utility")
    out = {k: torch.cat([p[k] for p in packs]).cuda() for k in fields}
    image, group, counts = torch.unique(out["image"], return_inverse=True, return_counts=True)
    receiver_image = torch.zeros(offset, dtype=torch.long, device="cuda")
    receiver_image.scatter_(0, out["receiver"], group)
    n_receivers = torch.bincount(receiver_image, minlength=len(image))[group]
    n_repeats = torch.bincount(out["receiver"], minlength=offset)[out["receiver"]]
    out.update(images=image, group=group, wg=(n_receivers * n_repeats).float().reciprocal(),
               wc=counts[group].float().reciprocal(),
               y_geo=(out["alignment"] >= .50).long() + (out["alignment"] >= .75).long(),
               y_cls=(out["supports"][:, 0] + 2 * out["supports"][:, 1]).long())
    assert (out["y_geo"][out["y_cls"] > 0] > 0).all()
    return out


def input_at(data, ix):
    receiver = data["receiver"][ix]
    return data["grid"][receiver], data["own"][receiver], data["box"][receiver], data["roles"][ix]


def pairs(train):
    pos = torch.nonzero(train["hard_fix"]).flatten().cpu().numpy()
    neg = torch.nonzero(train["hard_bad"]).flatten().cpu().numpy()
    assert (len(pos), len(neg)) == (895, 102)
    ids = train["image"].cpu().numpy()
    old = train["utility"].cpu().numpy()
    positive, same_count = [], 0
    for j in neg:
        same = pos[ids[pos] == ids[j]]
        same_count += int(len(same) > 0)
        pool = same if len(same) else pos
        positive.append(pool[np.abs(old[pool] - old[j]).argmin()])
    return torch.as_tensor(np.column_stack((positive, neg)), device="cuda"), same_count


def likelihood(geometry, state, data, ix):
    a = F.cross_entropy(geometry, data["y_geo"][ix], reduction="none")
    b = F.cross_entropy(state, data["y_cls"][ix], reduction="none")
    return a * data["wg"][ix], b * (data["y_geo"][ix] > 0) * data["wc"][ix]


@torch.inference_mode()
def measure(model, data):
    model.eval()
    a = b = 0.
    for ix in torch.arange(len(data["receiver"]), device="cuda").split(BATCH):
        terms = likelihood(*model(*input_at(data, ix)), data, ix)
        a += float(terms[0].sum())
        b += float(terms[1].sum())
    n = len(data["images"])
    return dict(total=(a + b) / n, geometry=a / n, role=b / n)


def bootstrap(train, cal):
    g = torch.bincount(train["y_geo"], weights=train["wg"], minlength=3)
    r = torch.bincount(train["y_cls"], weights=train["wc"] * (train["y_geo"] > 0), minlength=4)
    g = (g / g.sum()).log().expand(len(cal["image"]), -1)
    r = (r / r.sum()).log().expand(len(cal["image"]), -1)
    loss = sum(likelihood(g, r, cal, slice(None)))
    grouped = torch.zeros(len(cal["images"]), device="cuda").scatter_add_(0, cal["group"], loss).cpu().numpy()
    sample = np.random.default_rng(SEED).choice(grouped, (2000, len(grouped)), replace=True).mean(1)
    lo, hi = np.percentile(sample, (2.5, 97.5))
    return float(grouped.mean()), max(.005, float(hi - lo) * .5)


def save(model, names, kind, epoch, plan):
    path = ROOT / f"{NAME}_{kind}.pt"
    torch.save(dict(model=model.state_dict(), names=names, epoch=epoch, plan=plan), path.with_suffix(".part"))
    path.with_suffix(".part").replace(path)


def main():
    assert PLAN.is_file()
    plan = hashlib.sha256(PLAN.read_bytes()).hexdigest()
    result_path = ROOT / f"{NAME}_qualification.json"
    assert not result_path.exists()
    torch.manual_seed(SEED)
    torch.set_float32_matmul_precision("high")
    names, text = class_text()
    train, cal = load(("train", "mine")), load(("cal",))
    assert len(train["image"]) == 122079 and len(cal["image"]) == 14325
    assert set(train["images"].tolist()).isdisjoint(cal["images"].tolist())
    matched, same = pairs(train)
    baseline, tol = bootstrap(train, cal)
    model = ReceiverLocal(text).cuda()
    initial = measure(model, cal)
    opt = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.03)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS - 1, eta_min=.0001)
    n = len(train["receiver"])
    history = []
    best = math.inf
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for ix in torch.randperm(n, device="cuda").split(BATCH):
            opt.zero_grad(set_to_none=True)
            data_loss = sum(likelihood(*model(*input_at(train, ix)), train, ix)).mean() * n / len(train["images"])
            pair = matched[torch.randint(len(matched), (32,), device="cuda")]
            pair_score = utility(*model(*input_at(train, pair.flatten()))).reshape(-1, 2)
            rank_loss = F.softplus(2 * (pair_score[:, 1] - pair_score[:, 0])).mean()
            loss = data_loss + .10 * rank_loss
            assert torch.isfinite(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            opt.step()
        fit, valid = measure(model, train), measure(model, cal)
        record = dict(epoch=epoch, train=fit, cal=valid, lr=opt.param_groups[0]["lr"])
        assert all(math.isfinite(x) for table in (fit, valid) for x in table.values())
        history.append(record)
        if valid["total"] < best:
            best, best_epoch = valid["total"], epoch
            save(model, names, "best", epoch, plan)
        if epoch < EPOCHS:
            schedule.step()
        print(record, flush=True)
    save(model, names, "final", EPOCHS, plan)
    save_json(ROOT / f"{NAME}_history.json", history)
    tr, dv = (np.array([x[k]["total"] for x in history[-5:]]) for k in ("train", "cal"))
    plateau = abs(tr[:2].mean() - tr[-2:].mean()) / tr[:2].mean()
    slope = abs(np.polyfit(np.arange(5), dv, 1)[0])
    checks = dict(schedule=len(history) == EPOCHS and math.isclose(history[-1]["lr"], .0001), finite=True,
                  plateau=bool(plateau <= .025), slope=bool(slope <= tol / 4),
                  spread=bool(np.ptp(dv) <= tol), gap=bool(dv[-1] - best <= tol))
    result = dict(passed=all(checks.values()), checks=checks, plan=plan, initial_cal=initial, baseline_cal=baseline,
                  best_cal=best, best_epoch=best_epoch, final_cal=history[-1]["cal"], cal_tolerance=tol,
                  cal_slope=float(slope), plateau=float(plateau), paired=len(matched), same_image_pairs=same,
                  checkpoint=hashlib.sha256((ROOT / f"{NAME}_best.pt").read_bytes()).hexdigest(), eval_opened=False)
    save_json(result_path, result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
