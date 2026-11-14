"""Frozen Objects365 pilot for receiver-focused views and edge-preserving Qwen readout."""
import argparse
import math

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from admission_eval import frontier, passed, summarize
from layer_entropy_cache import encode, make_base
from layer_entropy_eval import entropy
from objects365_cache import WideGraphPack, block_path, eligible_actions, split_rows
from objects365_data import BLOCK, FOLDER, HERE, INDEX, SHARDS, digest, read, save_json
from objects365_fit import collect, qualified, replay_metrics

PLAN = HERE / "OBJECTS365_INSTANCE_GATE.md"
OUT = FOLDER / "instance_gate"
ARMS = ("original", "focus", "edge_focus")


class ReceiverAttention:
    def __init__(self, engine):
        self.native = engine.native
        attn = self.native.model.model.decoder.layers[-1].encoder_attn
        self.handle = attn.register_forward_hook(self.capture, with_kwargs=True)

    def capture(self, module, _args, kwargs, output):
        hidden = kwargs["hidden_states"]
        if kwargs.get("position_embeddings") is not None:
            hidden = hidden + kwargs["position_embeddings"]
        offsets = module.sampling_offsets(hidden).reshape(
            1, hidden.shape[1], module.n_heads, module.n_levels, module.n_points, 2)
        ref = kwargs["reference_points"][:, :, None, :, None]
        if ref.shape[-1] == 4:
            locations = ref[..., :2] + offsets / module.n_points * ref[..., 2:] * .5
        else:
            normalizer = kwargs["spatial_shapes"].flip(-1)[None, None, None, :, None, :]
            locations = ref + offsets / normalizer
        ratio = self.native.capture["valid_ratios"][:, None, None, :, None, :]
        self.points = (locations / ratio)[0].flatten(1, 3)
        self.weights = output[1][0].flatten(1)
        assert self.points.shape[:2] == self.weights.shape

    def crops(self, image, boxes, queries, reader):
        device = self.points.device
        boxes = boxes.to(device)
        span = boxes[:, 2:] - boxes[:, :2]
        assert (span > 0).all()
        points = (self.points[queries] - boxes[:, None, :2]) / span[:, None]
        inside = ((points >= 0) & (points <= 1)).all(-1)
        weights = self.weights[queries] * inside
        axis = (torch.arange(24, device=device) + .5) / 24
        y, x = torch.meshgrid(axis, axis, indexing="ij")
        xy = torch.stack((x, y), -1).reshape(1, 1, -1, 2)
        distance = (xy - points[:, :, None]).square().sum(-1)
        density = (torch.exp(-distance / (2 * .12**2)) * weights[:, :, None]).sum(1)
        strength = weights.sum(1) / self.weights[queries].sum(1).clamp_min(1e-8)
        density = density / density.amax(1, keepdim=True).clamp_min(1e-8)
        alpha = 1 - .8 * strength[:, None] * (1 - density)
        assert torch.isfinite(alpha).all()
        mean = torch.tensor(reader.processor.image_processor.image_mean).numpy() * 255
        dims = boxes.new_tensor([image.width, image.height] * 2)
        masked = []
        for coords, mask in zip((boxes * dims).tolist(), alpha.cpu()):
            x0, y0, x1, y1 = coords
            crop = image.crop((math.floor(x0), math.floor(y0), math.ceil(x1), math.ceil(y1)))
            # Resize the soft mask with the same physical support as the original crop.
            mask = F.interpolate(mask.reshape(1, 1, 24, 24), size=(crop.height, crop.width),
                                 mode="bilinear", align_corners=False)[0, 0].numpy()[..., None]
            pixels = np.asarray(crop, dtype=np.float32)
            masked.append(Image.fromarray(np.rint(mask * pixels + (1 - mask) * mean).clip(0, 255).astype(np.uint8)))
        features = []
        for start in range(0, len(masked), 32):
            p = reader.processor(images=masked[start:start+32], return_tensors="pt",
                                 input_data_format="channels_last").pixel_values.to(device, dtype=torch.float16)
            features.append(F.normalize(reader.clip.get_image_features(pixel_values=p).float(), dim=1))
        return torch.cat(features).cpu(), strength.cpu()


def edge_features(reader, pack, base, actions, focus):
    model = reader.graph.star
    device = model.texts.device
    lookup = {int(q): i for i, q in enumerate(pack["ids"])}
    result = []
    for start in range(0, len(actions), 64):
        q, c, slot = actions[start:start+64].T
        rows = torch.tensor([lookup[int(i)] for i in q])
        labels = pack["labels"][rows].to(device)
        text = model.texts[labels]
        valid = pack["valid"][rows].to(device)
        pair = torch.stack((base["label"][slot], c), dim=1).to(device)
        match = valid[:, None] & (labels[:, None] == pair[:, :, None])
        assert match[:, 1].any(1).all()
        donors = pack["features"][pack["j"][rows]].to(device).float()
        edge = model.edge(pack["stats"][rows].to(device).float()) * model.embedding_scale
        donor_tokens = model.instance(donors) * model.embedding_scale + text + edge
        read_tokens = model.read + text + edge
        # Native order and context for every real edge. Extra reads are used for classes without an edge.
        absent = ~match.any(-1)
        fallback = model.read + model.texts[pair]
        mask = torch.cat((torch.ones_like(valid[:, :1]), valid, valid, absent), dim=1)
        local = pack["features"][pack["i"][rows]].to(device).float()
        variants = torch.cat((local, torch.cat((local[:, :256], focus[start:start+len(q)].to(device)), dim=1)))
        shared = torch.cat((donor_tokens, read_tokens, fallback), dim=1).repeat(2, 1, 1)
        sequence = torch.cat(((model.instance(variants) * model.embedding_scale)[:, None], shared), dim=1)
        out = model.qwen(inputs_embeds=sequence.to(model.qwen.dtype), attention_mask=mask.long().repeat(2, 1),
                         output_hidden_states=True, use_cache=False)
        k = valid.shape[1]
        last = out.last_hidden_state[:, 1+k:].float()
        earlier = torch.stack([state[:, 1+k:].float() for state in out.hidden_states[1:-1]]).mean(0)
        hidden = torch.cat((last, earlier), dim=-1)
        pooled = torch.einsum("bck,bkd->bcd", match.float().repeat(2, 1, 1), hidden[:, :k])
        pooled = pooled / match.sum(-1).clamp_min(1).repeat(2, 1)[..., None]
        paired = torch.where(absent.repeat(2, 1)[..., None], hidden[:, k:], pooled).flatten(1)
        result.append(torch.stack(paired.half().cpu().chunk(2), dim=1))
    return torch.cat(result)


def probability(model, features):
    return torch.cat([model(x.float().cuda()).softmax(-1)[:, 1].cpu() for x in features.split(1024)])


@torch.inference_mode()
def features(doc, split, shard, limit=None):
    OUT.mkdir(exist_ok=True)
    model, receipt = qualified()
    engine = WideGraphPack(doc["names"], 365, doc["parents"])
    attention = ReceiverAttention(engine)
    index_hash = digest(INDEX)
    rows = split_rows(doc, split, shard)
    results = []
    for offset in range(0, len(rows), BLOCK):
        source = block_path(split, shard, offset//BLOCK)
        cache = torch.load(source, map_location="cpu", weights_only=True)
        assert cache["lock"] == dict(index=index_hash, images=[r["id"] for r in rows[offset:offset+BLOCK]])
        for record in cache["records"]:
            if "actions" not in record:
                continue
            frame = doc["rows"][record["image"]]
            actions = record["actions"]
            with Image.open(frame["path"]) as raw:
                image = raw.convert("RGB")
            original = probability(model, record["x"].flatten(0, 1)).reshape(-1, 3)
            ids, inverse = actions[:, 0].unique(return_inverse=True)
            for attempt in range(2):
                pack, _, _ = engine.native_pack(image)
                base = make_base(pack, 365)
                if split == "eval" and (not torch.equal(pack["boxes"], record["boxes"]) or
                                        not all(torch.equal(base[key], record["base"][key]) for key in base)):
                    continue
                keys = torch.as_tensor(eligible_actions(pack, base, 365)).reshape(-1, 4)
                if not torch.equal(keys[:, [0, 1, 3]], actions):
                    continue
                full = encode(engine.reader, pack, base, actions)[:, 0]
                error = float((full.float() - record["x"][:, 0].float()).abs().max())
                if error < .08:
                    break
            else:
                print("unreproduced", split, record["image"], flush=True)
                results.append(dict(image=record["image"], actions=actions, probability=original, skipped=True))
                continue
            focus, strength = attention.crops(image, pack["boxes"][ids], ids, engine.reader)
            paired = encode(engine.reader, pack, base, actions, [focus[inverse]])
            assert torch.equal(full, paired[:, 0])
            edge = edge_features(engine.reader, pack, base, actions, focus[inverse])
            old_focus = probability(model, paired[:, 1])
            edge = probability(model, edge.flatten(0, 1)).reshape(-1, 2)
            results.append(dict(image=record["image"], actions=actions,
                                probability=torch.column_stack((original, old_focus, edge)),
                                attention=strength[inverse], error=error, retries=attempt))
            if limit and len(results) >= limit:
                break
        print(split, shard, offset//BLOCK, len(results), flush=True)
        if limit and len(results) >= limit:
            break
    output = OUT / (f"smoke_{split}.pt" if limit else f"{split}_{shard}.pt")
    temp = output.with_suffix(".part")
    torch.save(dict(index=index_hash, checkpoint=receipt["checkpoint"], plan=digest(PLAN), rows=results), temp)
    temp.replace(output)


def scores(full, focus):
    views = torch.stack((full, focus), dim=1)
    mean = views.mean(1)
    js = (entropy(mean) - entropy(views).mean(1)).clamp_min(0)
    return ((2*mean-1) * (1-js/math.log(2))).masked_fill((views <= .5).any(1), -1), js


def picks(records, split, checkpoint, cap):
    by_image = {}
    for shard in range(SHARDS):
        cache = torch.load(OUT/f"{split}_{shard}.pt", map_location="cpu", weights_only=True)
        assert (cache["index"], cache["checkpoint"], cache["plan"]) == (digest(INDEX), checkpoint, digest(PLAN))
        by_image.update({r["image"]:r for r in cache["rows"]})
    assert len(by_image) == sum("actions" in r for r in records)
    result = {name:[] for name in ARMS}
    diagnostic = {name:dict(correct=0, harm=0, gain=0, unknown=0, first=0) for name in (*ARMS, "edge_only")}
    for record in records:
        if "actions" not in record:
            continue
        iid = record["image"]
        entry = by_image[iid]
        assert torch.equal(record["actions"], entry["actions"])
        original = entry["probability"][:, :3]
        old = original.mean(1)
        j = int(old.argmax())
        old_js = entropy(old[j]) - entropy(original[j]).mean()
        candidates = [("original", old, j, old_js)]
        if not entry.get("skipped"):
            a, _, _, d, e, f = entry["probability"].T
            focus, focus_js = scores(a,d)
            edge, edge_js = scores(e,f)
            candidates.append(("edge_only", e, int(e.argmax()), 0))
            for name, value, uncertainty in (("focus", focus, focus_js), ("edge_focus", edge, edge_js)):
                index = int(value.argmax())
                if value[index] > 0:
                    candidates.append((name, value, index, uncertainty[index]))
        for name, value, index, uncertainty in candidates:
            out = record["outcomes"][index]
            stat = diagnostic[name]
            stat["correct"] += out["correct"]
            stat["harm"] += bool(out["wrong"] or any(out["lost"]))
            stat["gain"] += bool(out["correct"] and any(out["new"]))
            stat["unknown"] += out["target"] < 0
            stat["first"] += 1
            if name == "edge_only" or (name == "original" and (value[index] <= .5 or uncertainty > cap)):
                continue
            if name != "original" and value[index] <= 0:
                continue
            q, c, slot = map(int, entry["actions"][index])
            result[name].append(dict(out, image=iid, query=q, proposed=c, slot=slot,
                                     score=float(value[index]), js=float(uncertainty)))
    parity = dict(skipped=[iid for iid, row in by_image.items() if row.get("skipped")],
                  retried=sum(row.get("retries",0) for row in by_image.values()),
                  max_error=max((row.get("error",0) for row in by_image.values()),default=0))
    return result, diagnostic, parity


def evaluate(doc):
    old = read(FOLDER/"policy.json")
    checkpoint = old["checkpoint"]
    policy_records,_ = collect(doc,"policy")
    policy_rows, policy_first, policy_parity = picks(policy_records,"policy",checkpoint,old["cap"])
    cutoffs = {name:frontier([r for r in rows if r["target"] >= 0],[],(0,),True)[0]
               for name,rows in policy_rows.items()}
    assert cutoffs["original"]["threshold"] == old["cutoffs"]["mixed_entropy"]["threshold"]
    save_json(OUT/"policy.json",dict(index=digest(INDEX),plan=digest(PLAN),checkpoint=checkpoint,cutoffs=cutoffs))
    records, counts = collect(doc,"eval")
    rows, first, parity = picks(records,"eval",checkpoint,old["cap"])
    selected = {name:[r for r in options if passed(r["score"],cutoffs[name]["threshold"])]
                for name,options in rows.items()}
    ap, annotated = replay_metrics(doc, records, selected)
    results = dict(images=counts["images"], annotated=annotated, policy_first=policy_first, first=first, cutoffs=cutoffs,
                   parity=dict(policy=policy_parity, eval=parity),
                   eligible={name:len(items) for name,items in rows.items()},
                   counts={name:summarize(items) for name,items in selected.items()},
                   unknown={name:sum(r["target"]<0 for r in items) for name,items in selected.items()},
                   gain_actions={name:sum(r["correct"] and any(r["new"]) for r in items) for name,items in selected.items()},
                   frontier={name:frontier(items,[],(0,1,2),True) for name,items in rows.items()},
                   harms={name:[r for r in items if r["wrong"] or any(r["lost"])] for name,items in selected.items()},
                   ap=ap)
    save_json(OUT/"result.json",results)
    print(results,flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode",choices=("policy","eval","smoke","score"))
    parser.add_argument("--shard",type=int,default=0)
    args = parser.parse_args()
    doc = read(INDEX)
    if args.mode == "score":
        evaluate(doc)
    else:
        features(doc,"policy" if args.mode=="smoke" else args.mode,args.shard,4 if args.mode=="smoke" else None)
