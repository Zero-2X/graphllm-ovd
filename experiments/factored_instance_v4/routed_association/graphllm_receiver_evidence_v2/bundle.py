"""Frozen stage construction and reload of the trained pipeline."""
import torch

from .common import OUT, load, locked_best, read
from .model import GraphReader


def construct(cfg,device,stage):
    torch.manual_seed(cfg["seed"])
    doc=read(OUT/"images_train.json")
    model=GraphReader(doc,cfg,device)
    if stage!="self":
        model.receiver.load_state_dict(locked_best("self")["state"],strict=True)
        model.add_evidence()
        if stage not in ("message",):
            model.evidence.load_state_dict(locked_best("message")["state"],strict=True)
            model.requires_grad_(False).eval()
    else:
        mean,std=load(OUT/"native_stats.pt")["values"]
        model.receiver.mean.copy_(mean)
        model.receiver.std.copy_(std)
    return model


