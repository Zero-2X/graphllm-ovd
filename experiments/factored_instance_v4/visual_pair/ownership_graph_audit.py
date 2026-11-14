"""Localize the failed safety tail using the qualified graph's cached evidence."""
import json

import numpy as np
import torch
from torch import nn

from dense_donor import ROOT, SHARDS, save_json
from ownership_graph_cache import EDGE_FIELDS, NODE_FIELDS
from ownership_graph_train import Readout, hypothesis_values, read, structure
from ownership_witness import screen


def auc(score, label):
    p, n = score[label == 1], np.sort(score[label == 0])
    if not len(p) or not len(n):
        return None
    strict = np.searchsorted(n, p, side="left")
    inclusive = np.searchsorted(n, p, side="right")
    return float((strict + inclusive).mean() / (2 * len(n)))


def load(name, model):
    receipt = json.loads((ROOT / f"{name}_qualification.json").read_text())
    state = torch.load(ROOT / f"{name}_best.pt", map_location="cpu", weights_only=True)
    assert receipt["passed"] and receipt["best_epoch"] == state["epoch"]
    model.load_state_dict(state["model"])
    return model.cuda().eval()


@torch.inference_mode()
def main():
    result = json.loads((ROOT / "ownership_graph_result.json").read_text())
    assert not result["passed"]
    index = json.loads((ROOT / "ownership_graph_index_receipt.json").read_text())
    cal = read(("cal",), index)
    evidence = load("ownership_graph_edges", nn.ModuleDict(dict(node=Readout(len(NODE_FIELDS), 2),
                                                              edge=Readout(len(EDGE_FIELDS), 2))))
    group = load("ownership_graph_group", nn.ModuleDict(dict(group=Readout(3, 3))))
    u, a, b, c, d = (x.cpu().numpy() for x in hypothesis_values(evidence, cal))
    f, active = structure(evidence, cal)
    p = group["group"](f, corrected=True).softmax(1).cpu().numpy()
    p[~active.cpu().numpy()] = (1., 0., 0.)
    archive = np.load(ROOT / "ownership_graph_development.npz")
    assert np.array_equal(cal["rows"], archive["rows"]) and np.allclose(p, archive["probability"], atol=1e-7)
    chosen = p.argmax(1)
    rows = cal["rows"]
    labels = rows[:, -1].astype(int)
    matrix = np.bincount(3 * labels + chosen, minlength=9).reshape(3, 3)
    group_id = cal["group"].cpu().numpy()
    edge_y = cal["edge_y"].cpu().numpy()
    assignment_y = cal["endpoint_y"].cpu().numpy()
    quality_y = cal["node_y"].cpu().numpy()
    links = cal["links"].cpu().numpy()
    link_group = group_id[links[:, 0]]
    in_distribution = cal["edge"][:, 0].cpu().numpy() > .5
    f = f[:, :2].sigmoid().cpu().numpy()
    lookup = {tuple(x[:14]): i for i, x in enumerate(rows)}
    fixed = np.concatenate([np.load(ROOT / f"joint_affine_probe_{i}.npz")["rows"] for i in range(SHARDS)])
    cases = []
    for row in fixed:
        i = lookup[tuple(row[:14])]
        node = np.flatnonzero(group_id == i)
        candidates = np.flatnonzero((link_group == i) & (d > 0))
        common = node[c[node].argmax()] if len(node) else None
        split = candidates[d[candidates].argmax()] if len(candidates) else None
        if split is not None:
            t, v = links[split]
            if a[t, 0] < a[t, 1]:
                t, v = v, t
            correct_split = bool(edge_y[split] == 0 and assignment_y[t, 0] == assignment_y[v, 1] == 1)
        else:
            correct_split = False
        cases.append(dict(image=int(row[0]), query=row[[1, 4]].astype(int).tolist(), state=int(row[7]),
                          box_iou=float(row[12]), score=float(archive["score"][i]), raw=float(f[i, 0] - f[i, 1]),
                          target=int(labels[i]), posterior=p[i].tolist(),
                          common_confirmed=None if common is None else bool(quality_y[common]),
                          common_correct=None if common is None else bool((assignment_y[common] == 1).all()),
                          selected_split_correct=correct_split))
    pos = sorted((r["score"] for r in cases if r["state"] == 1 and r["score"] > 0), reverse=True)
    errors = [r for r in cases if r["state"] == 2 and r["score"] >= pos[16]]
    different = [r for r in cases if r["state"] == 2]
    out = dict(witness_reliability_auc=auc(u, quality_y), endpoint_identity_auc=auc(a.flatten(), assignment_y.flatten()),
               witness_identity_auc=auc(1 - b, edge_y),
               witness_identity_auc_low_iou=auc(1 - b[~in_distribution], edge_y[~in_distribution]),
               witness_identity_auc_high_iou=auc(1 - b[in_distribution], edge_y[in_distribution]),
               group_confusion=matrix.tolist(), raw=screen(cases, "raw", True),
               correct_common_matched=sum(r["common_correct"] is True for r in cases if r["state"] == 1),
               correct_split_matched=sum(r["selected_split_correct"] for r in different),
               false_at17_without_confirmed_common=sum(r["common_confirmed"] is not True for r in errors),
               false_at17_target_unresolved=sum(r["target"] == 0 for r in errors),
               high=[r for r in different if r["box_iou"] >= .7])
    save_json(ROOT / "ownership_graph_audit.json", dict(out, cases=cases))
    print(out, flush=True)


if __name__ == "__main__":
    main()
