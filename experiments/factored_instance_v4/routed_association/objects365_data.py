"""Prepare a category-balanced Objects365 V2 train cohort from verified archives."""
from collections import Counter, defaultdict, deque
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile

from PIL import Image

HERE = Path(__file__).resolve().parent
BASE = Path(os.environ["BASE"])
ASSETS = BASE / "data/objects365_train"
FOLDER = HERE / "objects365"
INDEX = FOLDER / "manifest.json"
PLAN = HERE / "OBJECTS365_PLAN.md"
LIMIT = 8192
SHARDS = 8
BLOCK = 32

# Official category IDs remain the ground-truth IDs. These edits affect prompts.
PROMPTS = {
    "dinning table": "dining table", "moniter/tv": "television monitor",
    "liscense plate": "license plate", "cell phone": "mobile phone",
    "mouse": "computer mouse", "hotair balloon": "hot air balloon",
    "whiteboard": "whiteboard", "motorcycle": "motorcycle",
}
SHARED = {
    "tv": "moniter/tv", "mouse": "mouse", "remote": "remote",
    "motorcycle": "motorcycle", "frisbee": "frisbee", "skis": "skis",
}


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def save_json(path, obj):
    temp = path.with_suffix(".part")
    temp.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    temp.replace(path)


def annotation_file():
    output = ASSETS / "zhiyuan_objv2_train.json"
    if output.exists():
        return output
    with tarfile.open(ASSETS / "zhiyuan_objv2_train.tar.gz", "r|gz") as archive:
        for member in archive:
            if member.isfile() and Path(member.name).name == output.name:
                temp = output.with_suffix(".part")
                with archive.extractfile(member) as source, temp.open("wb") as file:
                    shutil.copyfileobj(source, file, length=1024 * 1024)
                assert temp.stat().st_size == member.size
                temp.replace(output)
                return output
    raise ValueError("Objects365 V2 train annotation member is missing")


def records(path, key):
    sys.path.insert(0, str(ASSETS / "pydeps"))
    import ijson

    with path.open("rb") as file:
        yield from ijson.items(file, key + ".item", use_float=True)


def categories(path):
    official = sorted(records(path, "categories"), key=lambda item: item["id"])
    assert len(official) == 365 and len({item["id"] for item in official}) == 365
    names = [PROMPTS.get(item["name"].strip().lower(), item["name"].strip().lower()) for item in official]
    assert len(set(names)) == 365, "duplicate prompt"
    previous = read(HERE.parent / "local_policy/manifest.json")
    by_source = {item["name"].strip().lower(): name for item, name in zip(official, names)}
    aligned = {label: by_source[SHARED.get(label, label)] for label in previous["base"]
               if SHARED.get(label, label) in by_source}
    parents_by_leaf = {aligned[leaf]: parent for leaf, parent in previous["parents"].items() if leaf in aligned}
    parents = sorted(set(parents_by_leaf.values()))
    parent_tokens = ["group of " + parent if parent in names else parent for parent in parents]
    parent_ids = {parent: len(names) + i for i, parent in enumerate(parents)}
    return dict(base=names, names=names+parent_tokens, categories={name: item["id"] for name, item in zip(names, official)},
                parents=[parent_ids.get(parents_by_leaf.get(name), -1) for name in names], shared_coco=aligned,
                official_categories=official)


def prepare(patch):
    FOLDER.mkdir(exist_ok=True)
    assert not INDEX.exists()
    path = annotation_file()
    doc = categories(path)
    save_json(FOLDER / "vocabulary.json", doc)
    print("vocabulary", len(doc["base"]), "exact shared", len(doc["shared_coco"]), flush=True)
    shard = ASSETS / f"patch{patch}.tar.gz"
    blocked_names = set()
    for path0 in (BASE / "data/omnilabel/annotations").glob("dataset_all_*.json"):
        blocked_names.update(Path(im["file_name"]).name for im in read(path0)["images"]
                             if im["file_name"].startswith("object365/"))
    with tarfile.open(shard, "r|gz") as archive:
        members = {Path(m.name).name for m in archive if m.isfile() and m.name.lower().endswith(".jpg")}
    eligible = members - blocked_names
    frames = {}
    for row in records(path, "images"):
        filename = Path(row["file_name"]).name
        if filename in eligible:
            frames[row["id"]] = dict(source_id=row["id"], file_name=filename,
                                     width=row["width"], height=row["height"], annotations=[])
    assert frames
    for ann in records(path, "annotations"):
        row = frames.get(ann["image_id"])
        if row is not None and min(ann["bbox"][2:]) > 0:
            row["annotations"].append(dict(id=ann["id"], bbox=ann["bbox"], category_id=ann["category_id"],
                                           iscrowd=int(ann.get("iscrowd", 0))))
    repeats = {}
    frequency = Counter()
    for iid, row in frames.items():
        count = Counter(a["category_id"] for a in row["annotations"] if not a["iscrowd"] and a["bbox"][2]*a["bbox"][3] >= 196)
        repeats[iid] = {c for c, n in count.items() if n >= 2}
        frequency.update(repeats[iid])
    print("patch rows", len(frames), "multi-instance", sum(bool(v) for v in repeats.values()),
          "repeated classes", len(frequency), flush=True)
    order = sorted(frames, key=lambda iid: hashlib.sha256(f"objects365-train-patch{patch}/{iid}".encode()).digest())
    buckets = defaultdict(deque)
    for iid in order:
        if repeats[iid]:
            owner = min(repeats[iid], key=lambda c: (frequency[c], c))
            buckets[owner].append(iid)
    selected = []
    while buckets and len(selected) < LIMIT:
        for c in sorted(buckets):
            selected.append(buckets[c].popleft())
            if not buckets[c]:
                del buckets[c]
            if len(selected) >= LIMIT:
                break
    requested = {frames[i]["file_name"]: frames[i] for i in selected}
    dest = ASSETS / "images" / f"patch{patch}"
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(shard, "r|gz") as archive:
        for member in archive:
            filename = Path(member.name).name
            if member.isfile() and filename in requested:
                output = dest / filename
                if not output.exists():
                    with archive.extractfile(member) as source:
                        temp = output.with_suffix(".part")
                        temp.write_bytes(source.read())
                        temp.replace(output)
    prior = read(HERE / "omni_expanded/manifest.json")
    seen = {row["sha"] for row in prior["rows"]}
    coco = read(HERE.parent / "local_policy/manifest.json")
    for row in coco["splits"]["cal_policy"]:
        image = HERE.parent / "local_policy/images" / row["dataset"] / row["file_name"]
        seen.add(digest(image))
    rows, counts = [], Counter()
    annotated = Counter()
    for iid in selected:
        row = frames[iid]
        output = dest / row["file_name"]
        with Image.open(output) as image:
            if image.size != (row["width"], row["height"]):
                counts["dimensions_rejected"] += 1
                continue
            image.verify()
        sha = digest(output)
        if sha in seen:
            counts["duplicate_bytes"] += 1
            continue
        seen.add(sha)
        bucket = int(sha[:8], 16) % 100
        split = "train" if bucket < 70 else "monitor" if bucket < 80 else "policy" if bucket < 90 else "eval"
        rows.append(dict(row, id=len(rows), path=str(output), sha=sha, split=split, known=list(doc["categories"].values())))
        annotated.update({a["category_id"] for a in row["annotations"]})
        counts[split] += 1
    sources = {p.name: (p.parent / (p.name + ".sha256")).read_text().split()[0]
               for p in (shard, ASSETS / "zhiyuan_objv2_train.tar.gz")}
    save_json(INDEX, dict(doc, rows=rows, counts=dict(counts), annotated_classes=len(annotated),
                          plan=digest(PLAN), official_split="train", archives=sources, eval_opened=False))
    print("cohort", dict(counts), "annotated classes", len(annotated), flush=True)


if __name__ == "__main__":
    prepare(int(sys.argv[1]))
