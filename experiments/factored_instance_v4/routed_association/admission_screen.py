"""Lock scores on natural calibration images before using any annotations."""
import argparse
import hashlib
import json

from PIL import Image
import torch

from route_screen import HERE, SHARDS, chunks, load, save_json, vocabulary, SemanticRuntime, ROOT, ONLINE_NAME
from receiver_admission_train import eligible, first_index, frozen, qualified, single_features, THRESHOLD


SPLIT = "route_train"
RESULT = HERE / "admission_calibration_lock.json"


def setup():
    path = HERE / "routing_manifest.json"
    manifest = json.loads(path.read_bytes())
    route_path = HERE / "routing_selection.json"
    route_raw = route_path.read_bytes()
    route = json.loads(route_raw)
    assert route["lock"]["manifest"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert route["lock"]["model"] == hashlib.sha256((ROOT / f"{ONLINE_NAME}_best.pt").read_bytes()).hexdigest()
    assert route["lock"]["threshold"] == THRESHOLD and not route["eval_opened"] and not route["labels_read"]
    assert all("annotations" not in r for rows in manifest["splits"].values() for r in rows)
    head, lock = qualified()
    lock.update(route=hashlib.sha256(route_raw).hexdigest(), split=SPLIT)
    return manifest, route, head, lock


def blocks(manifest, shard):
    for split, rows, source in chunks(manifest, shard):
        if split == SPLIT:
            yield rows, source, source.with_name("admission_" + source.name)


@torch.inference_mode()
def run(shard):
    manifest, route, head, lock = setup()
    names, nfine, parents = vocabulary(manifest, "cal")
    v2, reader, v2_names, prior = frozen()
    assert names[:nfine] == v2_names and prior == lock["prior"]
    runtime = SemanticRuntime(names, nfine, parents)
    complete = 0
    eligible_count = 0
    for rows, source, path in blocks(manifest, shard):
        if path.exists():
            records = load(path, rows, lock)
        else:
            assert hashlib.sha256(source.read_bytes()).hexdigest() == route["chunk_sha256"][source.name]
            historical = load(source, rows, route["lock"])
            records = []
            for row, old in zip(rows, historical):
                with Image.open(HERE / "images" / row["dataset"] / row["file_name"]) as src:
                    image = src.convert("RGB")
                pack, base, keys, _, utility, _ = runtime.encode(image)
                assert len(keys) == old["candidates"] and all(torch.equal(v, old["base"][k]) for k, v in base.items())
                record = dict(image=row["id"])
                if len(keys):
                    k = first_index(keys[:, 0], keys[:, 1], utility)
                    q, c, d, slot = map(int, keys[k])
                    u = float(utility[k])
                    prior_pick = old.get("selected")
                    assert (u >= THRESHOLD) == (prior_pick is not None)
                    if prior_pick is not None:
                        assert (q, c, d, slot) == tuple(prior_pick[k] for k in ("query", "proposed", "donor", "slot"))
                        assert abs(u - prior_pick["utility"]) <= .0001
                    if eligible(u):
                        x = single_features(v2, reader, pack, base, q, c, slot)
                        score = float(head.score(x[None])[0])
                        assert 0 < score < 1
                        record.update(query=q, proposed=c, slot=slot, old_utility=u, score=score,
                                      native={k: pack[k] for k in ("boxes", "logits")})
                else:
                    assert "selected" not in old
                records.append(record)
            torch.save(dict(lock=lock, records=records), path.with_suffix(".part"))
            path.with_suffix(".part").replace(path)
        complete += len(records)
        eligible_count += sum("score" in r for r in records)
        print(shard, complete, eligible_count, flush=True)


def finish():
    manifest, _, _, lock = setup()
    counts = 0
    hashes = {}
    ids = []
    for shard in range(SHARDS):
        for rows, _, path in blocks(manifest, shard):
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            records = load(path, rows, lock)
            ids.extend(r["image"] for r in records)
            counts += sum("score" in r for r in records)
    assert len(ids) == len(set(ids)) == 4096 and len(hashes) == 128 and counts > 0
    result = dict(lock=lock, images=len(ids), eligible=counts, chunk_sha256=hashes, labels_read=False, eval_opened=False)
    save_json(RESULT, result)
    print(result["images"], result["eligible"], "natural calibration scores locked", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--shard", type=int, choices=range(SHARDS))
    mode.add_argument("--finish", action="store_true")
    args = parser.parse_args()
    if args.finish:
        finish()
    else:
        run(args.shard)
