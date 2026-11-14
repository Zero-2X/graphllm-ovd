"""Complete the frozen foreground scores without rerunning GraphLLM proposals."""
import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image
import torch
from torch.nn import functional as F
from torchvision.ops import roi_align

from admission_screen import RESULT as CAL_LOCK, blocks, load, setup as admission_setup
from foreground_admission import HERE, NATURAL, OUTPUT as PANEL, SHARDS, ForegroundCrop, save_json, vocabulary
from receiver_admission_train import frozen, priors


PLAN = HERE / "FOREGROUND_ADMISSION_FULL_PLAN_ZH.md"
TARGETS = HERE / "foreground_full_targets.json"
RESULT = HERE / "foreground_full_lock.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def setup():
    manifest, route, head, prior = admission_setup()
    cal = json.loads(CAL_LOCK.read_bytes())
    natural = json.loads(TARGETS.read_bytes())
    panel = json.loads(PANEL.read_bytes())
    assert cal["lock"] == prior and not cal["labels_read"] and cal["eligible"] == 2850
    assert panel["lock"]["admission"] == {k: prior[k] for k in ("checkpoint", "plan", "prior")}
    assert not panel["labels_used"] and not natural["eval_opened"] and len(natural["targets"]) == 925
    assert natural["lock"]["manifest"] == digest(NATURAL / "manifest.json")
    lock = dict(calibration=digest(CAL_LOCK), panel=digest(PANEL), natural=digest(TARGETS), plan=digest(PLAN))
    return manifest, route, head, prior, cal, natural, panel, lock


class Scorer:
    def __init__(self, names, head):
        self.crop = ForegroundCrop(names)
        self.model, self.reader, classes, _ = frozen()
        assert names[:48] == classes
        self.head = head

    def score(self, image, pack, base, q, c, slot, check_grid=False):
        both, _ = self.crop.encode(image, pack, q)
        native = self.crop.native
        hidden = native.last_output.last_hidden_state[0, q:q + 1].float()
        own = torch.cat((F.normalize(hidden, dim=-1).expand(2, -1).half(), both.cuda()), dim=1)
        h, w = native.capture["spatial_shapes"][0].tolist()
        visual = native.capture["vision_encoder_hidden_states"][0, :h*w].reshape(h, w, -1).permute(2, 0, 1)[None].contiguous()
        scale = (native.capture["valid_ratios"][0, 0] * visual.new_tensor([w, h])).repeat(2)
        box = pack["boxes"][q:q + 1].cuda()
        pad = .25 * (box[:, 2:] - box[:, :2])
        roi = (box + torch.cat((-pad, pad), dim=1)).clamp(0, 1)
        grid = roi_align(visual, [roi * scale], 5, aligned=True).flatten(2).transpose(1, 2).half()
        if check_grid:
            index = torch.nonzero(pack["ids"] == q).flatten()
            assert torch.equal(grid.cpu(), pack["grid"][index])
            cached = pack["features"][pack["i"][index], :256].cuda()
            error = float((own[:1, :256].float() - cached.float()).abs().max())
            assert error <= .001, (q, error)
            if error:
                print("reused exact cached receiver vector", q, error, flush=True)
            own[:, :256] = cached.expand(2, -1)
        old = int(base["label"][slot])
        assert int(base["query"][slot]) == q and old != c
        roles = torch.tensor([[old, c]], device="cuda").expand(2, -1)
        logits = pack["logits"][q:q+1, [old, c]].cuda().expand(2, -1)
        evidence = self.reader.extract(grid.expand(2, -1, -1), own, box.expand(2, -1), roles,
                                       torch.arange(2, device="cuda"), logits)
        score = self.head.score(priors(self.model, *evidence).cpu()).tolist()
        assert all(0 < v < 1 for v in score)
        return score


def natural_record(row, natural):
    path = NATURAL / row["chunk"]
    assert Path(row["chunk"]).name == row["chunk"]
    data = torch.load(path, map_location="cpu", weights_only=True)
    assert data["lock"] == natural["lock"]
    return next(r for r in data["records"] if r["image"] == row["image"])


def from_image(path, scorer, pack, base, record, check_grid=False):
    with Image.open(path) as source:
        image = source.convert("RGB")
    keys = (record[k] for k in ("query", "proposed", "slot"))
    return scorer.score(image, pack, base, *keys, check_grid=check_grid)


@torch.inference_mode()
def run(shard):
    manifest, route, head, prior, cal, natural, panel, lock = setup()
    destination = HERE / f"foreground_full_{shard}.json"
    if destination.is_file():
        saved = json.loads(destination.read_bytes())
        assert saved["lock"] == lock
        print("reused complete shard", shard, len(saved["calibration"]), len(saved["natural"]), flush=True)
        return
    names, _, _ = vocabulary(manifest, "cal")
    scorer = Scorer(names, head)
    if shard == 0:
        probes = {r["image"]: r for r in panel["outputs"] if r["image"] in (151831, 219268)}
        for row in natural["targets"]:
            if row["image"] not in probes:
                continue
            record = natural_record(row, natural)
            score = from_image(NATURAL / "images" / row["dataset"] / row["file_name"], scorer,
                               record["pack"], record["base"], row, True)
            assert abs(score[1] - probes[row["image"]]["scores"][1]) <= .00001
        assert len(probes) == 2
    cal_outputs = []
    for j, (rows, source, path) in enumerate(blocks(manifest, shard)):
        assert digest(path) == cal["chunk_sha256"][path.name] and digest(source) == route["chunk_sha256"][source.name]
        assert len(rows) == 32
        store = HERE / f"foreground_cal_{shard}_{j:03d}.json"
        if store.exists():
            saved = json.loads(store.read_bytes())
            assert saved["lock"] == lock
            outputs = saved["outputs"]
        else:
            outputs = []
            for row, prior_record, native_record in zip(rows, load(path, rows, prior), load(source, rows, route["lock"])):
                if "score" not in prior_record:
                    continue
                score = from_image(HERE / "images" / row["dataset"] / row["file_name"], scorer,
                                   prior_record["native"], native_record["base"], prior_record)
                outputs.append(dict(image=row["id"], prior=prior_record["score"], check=score[0], score=score[1]))
            save_json(store, dict(lock=lock, outputs=outputs))
        cal_outputs.extend(outputs)
        if j % 4 == 3:
            print("calibration", shard, (j + 1) * 32, len(cal_outputs), flush=True)

    used = {r["image"] for r in panel["outputs"]}
    targets = [r for r in natural["targets"][shard::SHARDS] if r["image"] not in used]
    outputs = []
    last_chunk = None
    for row in targets:
        if row["chunk"] != last_chunk:
            last_chunk = row["chunk"]
            data = torch.load(NATURAL / last_chunk, map_location="cpu", weights_only=True)
            assert data["lock"] == natural["lock"]
            cache = {x["image"]: x for x in data["records"]}
        record = cache[row["image"]]
        result = from_image(NATURAL / "images" / row["dataset"] / row["file_name"], scorer,
                            record["pack"], record["base"], row, True)
        outputs.append(dict(image=row["image"], prior=row["score"], check=result[0], score=result[1]))
        if len(outputs) % 32 == 0 or len(outputs) == len(targets):
            print("natural", shard, len(outputs), len(targets), flush=True)
    save_json(destination, dict(lock=lock, calibration=cal_outputs, natural=outputs))


def finish():
    _, _, _, _, cal, natural, panel, lock = setup()
    records = dict(calibration=[], natural=[])
    for shard in range(SHARDS):
        data = json.loads((HERE / f"foreground_full_{shard}.json").read_bytes())
        assert data["lock"] == lock
        for k in records:
            records[k].extend(data[k])
    targets = {r["image"]: r["score"] for r in natural["targets"]}
    records["natural"].extend(dict(image=r["image"], prior=targets[r["image"]], check=r["scores"][0], score=r["scores"][1])
                              for r in panel["outputs"] if r["image"] in targets)
    assert len({r["image"] for r in records["natural"]}) == len(records["natural"]) == len(targets)
    assert len({r["image"] for r in records["calibration"]}) == len(records["calibration"]) == cal["eligible"]
    assert set(targets) == {r["image"] for r in records["natural"]}
    errors = {k: max(abs(r["check"] - r["prior"]) for r in v) for k, v in records.items()}
    assert all(v <= .01 for v in errors.values()), errors
    save_json(RESULT, dict(lock=lock, **records, old_score_max_error=errors, labels_used=False, eval_opened=False))
    print({k: len(v) for k, v in records.items()}, errors, flush=True)


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
