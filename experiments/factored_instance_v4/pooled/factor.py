"""Separate receiver alignment, conditional label evidence and action risk."""
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

V3 = Path(__file__).resolve().parents[2] / "utility_gate_v3"
sys.path.insert(0, str(V3))
from counterfactual import candidate_actions
from gate import FrozenEvidence
from paired_targets import KINDS, LEVELS


def owner_slots(pack, base):
    owner = torch.full((len(pack["boxes"]),), -1, dtype=torch.long)
    for i, q in enumerate(base["query"].tolist()):
        if owner[q] < 0:
            owner[q] = i
    return owner


class FactorEvidence(FrozenEvidence):
    @torch.inference_mode()
    def spatial(self, pack, r):
        q = pack["ids"][r]
        xyxy = pack["boxes"][q].float()
        wh = (xyxy[:, 2:] - xyxy[:, :2]).clamp_min(1e-6)
        own = pack["features"][pack["i"][r], :256]
        geometry = torch.cat((xyxy, wh, wh.prod(-1, keepdim=True)), dim=-1)
        return pack["grid"][r].reshape(len(r), 25, 256).half(), torch.cat((own, geometry), dim=-1).half()

    @torch.inference_mode()
    def claims(self, pack, base, r, e, owner):
        q = pack["ids"][r]
        c = pack["labels"][r, e]
        old = owner[q]
        old_label = pack["logits"][q, :self.nfine].argmax(-1)
        if len(base["label"]):
            old_label = torch.where(old >= 0, base["label"][old.clamp_min(0)], old_label)
        prop = dict(receiver=r, edge=e, query=q, label=c, old=old, old_label=old_label, box=pack["boxes"][q])
        graph = self.graph.encode(pack, prop).float()
        fit = pack["features"][pack["i"][r], 256:].float() @ self.text.T
        step = torch.arange(len(q))
        old_fit = fit[step, old_label]
        new_fit = fit[step, c]
        sibling = (self.parents[None] == self.parents[c, None]) & (self.parents[c, None] >= 0)
        sibling[step, c] = False
        best = fit.amax(-1)
        other = torch.where(sibling.any(-1), fit.masked_fill(~sibling, -1).amax(-1), best)
        same = c == old_label
        parent = (self.parents[c] == self.parents[old_label]) & (self.parents[c] >= 0)
        local = torch.stack((old_fit, new_fit, new_fit-old_fit, new_fit-other, new_fit-best,
                             parent.float(), (~same).float()), dim=-1)
        x = torch.cat((graph, local), dim=-1).half()
        assert torch.isfinite(x).all()
        return x, same

    @torch.inference_mode()
    def encode(self, pack, base, extra_claims=(), extra_receivers=()):
        actions = [(a, score) for score in LEVELS for a in candidate_actions(pack, base, self.nfine, score)]
        if not actions and not len(extra_claims) and not len(extra_receivers):
            return actions, None
        owner = owner_slots(pack, base)
        r, e = torch.nonzero(pack["valid"] & (pack["labels"] < self.nfine) & ~pack["high"][:, None], as_tuple=True)
        first = {}
        for i, j in zip(r.tolist(), e.tolist()):
            first.setdefault((int(pack["ids"][i]), int(pack["labels"][i, j])), (i, j))
        action_edges = [first[q, c] for (_, q, c, _), _ in actions]
        claims = list(dict.fromkeys([*action_edges, *extra_claims]))
        receivers = list(dict.fromkeys([*(i for i, _ in claims), *extra_receivers]))
        r_index = {r: i for i, r in enumerate(receivers)}
        c_index = {edge: i for i, edge in enumerate(claims)}
        grid, own = self.spatial(pack, torch.tensor(receivers, dtype=torch.long))
        if claims:
            r, e = torch.tensor(claims).unbind(1)
            x, same = self.claims(pack, base, r, e, owner)
        else:
            x, same = torch.empty((0, self.width+22), dtype=torch.float16), torch.empty(0, dtype=torch.bool)
        meta = []
        for (kind, q, _, slot), score in actions:
            own_slot = int(owner[q])
            current = float(base["score"][own_slot]) if own_slot >= 0 else 0.
            displaced = float(base["score"][slot]) if slot is not None else 0.
            meta.append([*(float(kind == k) for k in KINDS), score, current, displaced,
                         score-current, len(base["score"])/100, float(slot is not None)])
        return actions, dict(grid=grid, own=own, x=x, same=same, r_index=r_index, c_index=c_index,
                             rec=torch.tensor([r_index[r] for r, _ in action_edges], dtype=torch.long),
                             claim=torch.tensor([c_index[k] for k in action_edges], dtype=torch.long),
                             meta=torch.tensor(meta, dtype=torch.float16).reshape(-1, 10))


class FactoredGate(nn.Module):
    def __init__(self, dim, context_dim):
        super().__init__()
        self.context_dim = context_dim
        self.alignment = nn.Linear(3*256+7, 2)
        self.context = nn.LayerNorm(context_dim)
        self.semantic = nn.Sequential(nn.Linear(dim, 128), nn.GELU(), nn.Linear(128, 5))
        self.effect = nn.Sequential(nn.Linear(10+3+4, 64), nn.GELU(), nn.Linear(64, 4))

    def alignment_logits(self, grid, own):
        grid = grid.float().reshape(-1, 5, 5, 256)
        center = grid[:, 1:4, 1:4].sum((1, 2))
        ring = grid.sum((1, 2))-center
        inside = F.normalize(center, dim=-1)
        outside = F.normalize(ring, dim=-1)
        query = F.normalize(own[:, :256].float(), dim=-1)
        features = torch.cat((16*inside, 16*(inside-outside), 16*query, own[:, -7:].float()), dim=-1)
        q50, conditional75 = self.alignment(features).unbind(-1)
        yes = F.logsigmoid(q50)
        return torch.stack((F.logsigmoid(-q50), yes+F.logsigmoid(-conditional75),
                            yes+F.logsigmoid(conditional75)), dim=-1)

    def semantic_logits(self, x, same):
        x = x.float()
        z = self.semantic(torch.cat((self.context(x[:, :self.context_dim]), x[:, self.context_dim:]*2), dim=-1))
        allowed = torch.stack((torch.ones_like(same), ~same, ~same, same), dim=-1)
        return z[:, :4].masked_fill(~allowed, -1e4), z[:, 4]

    def effect_logits(self, quality, semantic, meta):
        return self.effect(torch.cat((quality, semantic, meta.float()), dim=-1))

    @torch.inference_mode()
    def probabilities(self, pack):
        q = self.alignment_logits(pack["grid"].cuda(), pack["own"].cuda()).softmax(-1)[pack["rec"].cuda()]
        s, donor = self.semantic_logits(pack["x"].cuda(), pack["same"].cuda())
        s = s.softmax(-1)[pack["claim"].cuda()]
        raw = self.effect_logits(q, s, pack["meta"].cuda())
        outcome = raw[:, :3].softmax(-1)
        old, new = s[:, 1]+s[:, 3], s[:, 2]+s[:, 3]
        intent = q[:, 1:].sum(-1)*new
        safe = intent*outcome[:, 1]
        harm = intent*outcome[:, 2]+(1-intent)*raw[:, 3].sigmoid()
        return safe-4*harm, torch.stack((old, new, donor.sigmoid()[pack["claim"].cuda()],
                                        q[:, 1:].sum(-1), q[:, 2], safe, harm), dim=-1)


def read_factor(path="best.pt"):
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = FactoredGate(state["dim"], state["context_dim"]).cuda().eval()
    model.load_state_dict(state["model"])
    return model


@torch.inference_mode()
def select_factor(actions, evidence, base, model):
    if not actions:
        return None
    utility, p = model.probabilities(evidence)
    owner = {}
    for q, c, score in zip(base["query"].tolist(), base["label"].tolist(), base["score"].tolist()):
        owner.setdefault(q, (c, score))
    changes_class = torch.tensor([q in owner and c != owner[q][0] and
                                  (kind == "replace" or score > owner[q][1])
                                  for (kind, q, c, _), score in actions], device=utility.device)
    allowed = ~changes_class | (p[:, 1] > p[:, 0])
    if not allowed.any():
        return None
    i = int(utility.masked_fill(~allowed, -torch.inf).argmax())
    return i, float(utility[i]), p[i].tolist()
