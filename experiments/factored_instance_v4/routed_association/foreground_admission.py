"""Test class evidence restricted to the original receiver query's foreground."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from transformers import AutoProcessor, CLIPModel

from route_screen import HERE, ROOT, SHARDS, save_json, vocabulary
from natural_funnel import HERE as NATURAL
from receiver_admission_train import NAME, first_index, frozen, qualified, single_features
from donor_transport_probe import Native
from ownership_graph_audit import load as load_head
from pixel_head_train import PixelHead


PLAN = HERE / "FOREGROUND_ADMISSION_PLAN_ZH.md"
MANIFEST = HERE / "foreground_admission_targets.json"
OUTPUT = HERE / "foreground_admission_lock.json"
GRID = 32


class ForegroundCrop:
    def __init__(self, names):
        self.native = Native(names)
        self.mask = load_head("pixel_head", PixelHead())
        model = Path(os.environ["BASE"]) / "models" / "clip"
        self.processor = AutoProcessor.from_pretrained(model, local_files_only=True)
        self.clip = CLIPModel.from_pretrained(model, local_files_only=True, torch_dtype=torch.float16,
                                              use_safetensors=True).cuda().eval()
        self.mean = np.asarray(self.processor.image_processor.image_mean, dtype=np.float32) * 255

    def crop(self, image, box, field, q):
        w, h = image.size
        x0, y0, x1, y1 = (box * box.new_tensor([w, h, w, h])).tolist()
        left, top, right, bottom = math.floor(x0), math.floor(y0), math.ceil(x1), math.ceil(y1)
        assert right > left and bottom > top
        crop = image.crop((left, top, right, bottom))
        axis = (torch.arange(GRID, device="cuda", dtype=torch.float32) + .5) / GRID
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        points = torch.stack(((left + xx * crop.width) / w, (top + yy * crop.height) / h), dim=-1).reshape(1, -1, 2)
        ratio = self.native.capture["valid_ratios"][0, 0]
        pixels = F.grid_sample(field, (2 * points * ratio - 1)[None], align_corners=False)[0].permute(1, 2, 0)
        relative = (points - (box[:2] + box[2:]) * .5) / (box[2:] - box[:2]).clamp_min(1e-6)
        hidden = self.native.last_output.last_hidden_state[0, q:q + 1]
        mask = self.mask(hidden.half(), pixels.half(), relative.half()).sigmoid().reshape(1, 1, GRID, GRID)
        alpha = F.interpolate(mask, size=(crop.height, crop.width), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
        pixels = np.asarray(crop, dtype=np.float32)
        foreground = np.rint(pixels * alpha[..., None] + self.mean * (1 - alpha[..., None])).clip(0, 255).astype(np.uint8)
        return crop, Image.fromarray(foreground), dict(mean=float(mask.mean()), min=float(mask.min()), max=float(mask.max()))

    @torch.inference_mode()
    def encode(self, image, pack, q):
        boxes, logits, field, _ = self.native.forward(image)
        assert torch.equal(boxes.cpu(), pack["boxes"]) and torch.equal(logits.cpu(), pack["logits"])
        original, foreground, info = self.crop(image, boxes[q], field, q)
        data = self.processor(images=[original, foreground], return_tensors="pt").to("cuda")
        feature = F.normalize(self.clip.get_image_features(**data).float(), dim=-1).half().cpu()
        assert feature.isfinite().all()
        return feature, info


def setup():
    raw = MANIFEST.read_bytes()
    spec = json.loads(raw)
    assert not spec["eval_opened"] and len(spec["targets"]) == len({x["image"] for x in spec["targets"]})
    assert len(spec["targets"]) > 150 and "annotations" not in spec
    assert spec["lock"]["manifest"] == hashlib.sha256((NATURAL / "manifest.json").read_bytes()).hexdigest()
    head, admission = qualified()
    assert spec["admission"] == admission["checkpoint"]
    lock = dict(targets=hashlib.sha256(raw).hexdigest(), plan=hashlib.sha256(PLAN.read_bytes()).hexdigest(),
                pixel_head=hashlib.sha256((ROOT / "pixel_head_best.pt").read_bytes()).hexdigest(), admission=admission)
    return spec, head, lock


@torch.inference_mode()
def run(shard):
    spec, head, lock = setup()
    manifest = {k: spec[k] for k in ("base", "novel", "parents", "categories")}
    names, nfine, _ = vocabulary(manifest, "cal_policy")
    model, reader, fine_names, prior = frozen()
    assert nfine == 48 and names[:nfine] == fine_names and prior == lock["admission"]["prior"]
    crop = ForegroundCrop(names)
    targets = spec["targets"][shard::SHARDS]
    group = {}
    for row in targets:
        group.setdefault(row["chunk"], []).append(row)
    outputs = []
    for file, rows in group.items():
        assert Path(file).name == file
        data = torch.load(NATURAL / file, map_location="cpu", weights_only=True)
        assert data["lock"] == spec["lock"]
        records = {r["image"]: r for r in data["records"]}
        for row in rows:
            record = records[row["image"]]
            pack, base, keys = (record[k] for k in ("pack", "base", "keys"))
            old_u = record["utility"].numpy()
            k = first_index(keys[:, 0].numpy(), keys[:, 1].numpy(), old_u)
            q, c, _, slot = map(int, keys[k])
            assert (q, c, slot) == tuple(row[k] for k in ("query", "proposed", "slot"))
            with Image.open(NATURAL / "images" / row["dataset"] / row["file_name"]) as raw:
                image = raw.convert("RGB")
            both, mask = crop.encode(image, pack, q)
            r = torch.nonzero(pack["ids"] == q).flatten()
            assert len(r) == 1
            node = int(pack["i"][r[0]])
            prior_clip = pack["features"][node, 256:]
            err = float((both[0].float() - prior_clip.float()).abs().max())
            assert err <= .003, (row["image"], err)
            cls = [int(base["label"][slot]), c]
            clip = F.normalize(torch.stack((prior_clip, both[1])).cuda().float(), dim=1) @ reader.text[cls].T
            scores = []
            for vector in (prior_clip, both[1]):
                local = pack["features"].clone()
                local[node, 256:] = vector
                x = single_features(model, reader, dict(pack, features=local), base, q, c, slot)
                scores.append(float(head.score(x[None])[0]))
            outputs.append(dict(image=row["image"], query=q, proposed=c, slot=slot, scores=scores,
                                fits=clip.cpu().tolist(), mask=mask, max_clip_error=err))
            if len(outputs) % 10 == 0 or len(outputs) == len(targets):
                print(shard, len(outputs), len(targets), flush=True)
    save_json(HERE / f"foreground_admission_{shard}.json", dict(lock=lock, outputs=outputs))


def finish():
    spec, _, lock = setup()
    outputs = []
    for s in range(SHARDS):
        data = json.loads((HERE / f"foreground_admission_{s}.json").read_text())
        assert data["lock"] == lock
        outputs.extend(data["outputs"])
    key = lambda x: tuple(x[k] for k in ("image", "query", "proposed", "slot"))
    assert sorted(map(key, outputs)) == sorted(map(key, spec["targets"]))
    save_json(OUTPUT, dict(lock=lock, outputs=outputs, labels_used=False, eval_opened=False))
    print("foreground evidence locked", len(outputs), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--shard", type=int, choices=range(SHARDS))
    mode.add_argument("--finish", action="store_true")
    a = parser.parse_args()
    if a.finish:
        finish()
    else:
        run(a.shard)
