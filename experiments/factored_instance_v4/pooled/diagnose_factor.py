"""Inspect the development-set factors without opening the independent Eval."""
from pathlib import Path

import numpy as np
import torch

from factor import read_factor
from counterfactual import save_json


def auc(y, score):
    y = np.asarray(y, dtype=bool)
    score = np.asarray(score)
    assert y.any() and (~y).any()
    scores, inverse, counts = np.unique(score, return_inverse=True, return_counts=True)
    lower = counts.cumsum()-counts
    rank = lower[inverse]+(counts[inverse]+1)/2
    return float((rank[y].sum()-y.sum()*(y.sum()+1)/2)/(y.sum()*(~y).sum()))


@torch.inference_mode()
def main():
    data = torch.load("factors_cal.pt", map_location="cpu", weights_only=True)
    model = read_factor()
    quality = []
    for r in data["q"][:, 0].split(1024):
        quality.append(model.alignment_logits(data["grid"][r].cuda(), data["own"][r].cuda()).softmax(-1).cpu())
    q = torch.cat(quality)
    y = data["q"][:, 1]
    alignment = {}
    for cut in (1, 2):
        alignment[str(50 if cut==1 else 75)] = dict(n=len(y), positives=int((y>=cut).sum()),
            factor_auc=auc(y>=cut,q[:,cut:].sum(-1)), native_auc=auc(y>=cut,data["raw_conf"]))
    claim = data["s"][:, 0]
    values = []
    for idx in claim.split(1024):
        z, _ = model.semantic_logits(data["x"][idx].cuda(), data["same"][idx].cuda())
        values.append(z.softmax(-1).cpu())
    p = torch.cat(values)
    states = data["s"][:, 1]
    decisive = (states==1)|(states==2)
    choose_new = p[:,2]>p[:,1]
    native = data["x"][claim, data["context_dim"]+10].float()
    semantics = dict(n=len(states), old_only=int((states==1).sum()), new_only=int((states==2).sum()),
                     choose_new_when_correct=int((choose_new & (states==2)).sum()),
                     choose_new_when_wrong=int((choose_new & (states==1)).sum()),
                     decisive_factor_auc=auc(states[decisive]==2,(p[:,2]-p[:,1])[decisive]),
                     decisive_native_auc=auc(states[decisive]==2,native[decisive]))
    result = dict(checkpoint="best.pt", split="v1 development Cal", alignment=alignment, semantics=semantics)
    if Path("policy.json").exists():
        import json
        policy = json.loads(Path("policy.json").read_text())
        rows = policy["thresholds"]
        fp_safe = [r for r in rows if r["fp_delta_035"]<=.10*max(0,r["new_035"]-r["lost_035"])]
        result["policy_frontier"] = dict(max_valid_fraction=max((r["valid"]/r["edits"] for r in rows),default=0),
                                         max_strict_under_fp_budget=max((r["strict_gain"] for r in fp_safe),default=0))
    save_json("diagnosis.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    main()