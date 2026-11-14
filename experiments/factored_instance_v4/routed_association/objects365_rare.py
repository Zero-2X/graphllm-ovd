"""Fill missing Objects365 train category coverage from additional official shards."""
from collections import Counter, defaultdict
import hashlib
from pathlib import Path
import subprocess
import tarfile
import urllib.request

from PIL import Image

from objects365_data import ASSETS, BASE, FOLDER, HERE, INDEX, digest, read, records, save_json


RARE_PLAN = HERE / "OBJECTS365_RARE_PLAN.md"
RARE_INDEX = FOLDER / "rare_manifest.json"
SOURCE = "https://dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365%E6%95%B0%E6%8D%AE%E9%9B%86/train/"
PER_CLASS = 24


def archive_for(patch):
    output = ASSETS / f"{patch}.tar.gz"
    if output.is_file():
        return output
    url = SOURCE + output.name
    request = urllib.request.Request(url, headers={"Range": "bytes=0-31"})
    with urllib.request.urlopen(request, timeout=30) as response:
        assert response.status == 206
        length = int(response.headers["Content-Range"].rsplit("/", 1)[1])
    temp = output.with_suffix(".part")
    print("download", output.name, length, flush=True)
    subprocess.run(["curl", "-fsSL", "--retry", "5", "--connect-timeout", "15", "-C", "-",
                    "-o", str(temp), url], check=True)
    assert temp.stat().st_size == length
    temp.replace(output)
    (ASSETS / (output.name+".sha256")).write_text(digest(output)+"  "+output.name+"\n")
    return output


def main():
    assert not RARE_INDEX.exists()
    main_index = read(INDEX)
    covered = {a["category_id"] for row in main_index["rows"] if row["split"]=="train" for a in row["annotations"]}
    missing = set(main_index["categories"].values()) - covered
    annotation_path = ASSETS / "zhiyuan_objv2_train.json"
    targets = defaultdict(Counter)
    for ann in records(annotation_path, "annotations"):
        if ann["category_id"] in missing and not ann.get("iscrowd", 0) and min(ann["bbox"][2:])>0:
            targets[ann["image_id"]][ann["category_id"]]+=1
    print("rare targets", len(missing), "annotated image IDs", len(targets), flush=True)
    blocked = {row["file_name"] for row in main_index["rows"]}
    for path in (BASE / "data/omnilabel/annotations").glob("dataset_all_*.json"):
        blocked.update(Path(row["file_name"]).name for row in read(path)["images"]
                       if row["file_name"].startswith("object365/"))
    by_patch = defaultdict(lambda: defaultdict(list))
    frames = {}
    for image in records(annotation_path, "images"):
        iid = image["id"]
        if iid not in targets:
            continue
        parts = Path(image["file_name"]).parts
        if parts[-1] in blocked:
            continue
        patch = parts[-2]
        if not patch.startswith("patch") or not patch[5:].isdigit():
            continue
        frames[iid] = dict(source_id=iid, file_name=parts[-1], width=image["width"], height=image["height"],
                           patch=patch, annotations=[])
        for c in targets[iid]:
            by_patch[patch][c].append(iid)
    selected, shards, remaining = set(), [], set(missing)
    while remaining and by_patch:
        patch = max(sorted(by_patch),key=lambda p:(len(remaining & by_patch[p].keys()),
                    sum(min(PER_CLASS, len(by_patch[p].get(c, ()))) for c in remaining)))
        avail = by_patch.pop(patch)
        found = remaining & avail.keys()
        if not found:
            break
        shards.append(patch)
        for c in sorted(found):
            choices = sorted(avail[c], key=lambda iid:(-int(targets[iid][c]>=2),
                              hashlib.sha256(f"objects365-rare/{c}/{iid}".encode()).digest()))
            selected.update(choices[:PER_CLASS])
        remaining -= found
    print("rare selection", len(selected), "shards", shards, "classes", len(missing)-len(remaining), flush=True)
    source_receipts = {}
    for patch in shards:
        archive_path = archive_for(patch)
        source_receipts[archive_path.name] = (ASSETS / (archive_path.name+".sha256")).read_text().split()[0]
        dest = ASSETS / "images" / patch
        dest.mkdir(parents=True, exist_ok=True)
        wanted = {frames[i]["file_name"] for i in selected if frames[i]["patch"]==patch}
        with tarfile.open(archive_path, "r|gz") as archive:
            for member in archive:
                name = Path(member.name).name
                if member.isfile() and name in wanted:
                    output = dest / name
                    if not output.exists():
                        with archive.extractfile(member) as source:
                            temp = output.with_suffix(".part")
                            temp.write_bytes(source.read())
                            temp.replace(output)
        print("rare shard extracted", patch, len(wanted), flush=True)
    for ann in records(annotation_path, "annotations"):
        iid = ann["image_id"]
        if iid in selected and min(ann["bbox"][2:])>0:
            frames[iid]["annotations"].append(dict(id=ann["id"], bbox=ann["bbox"], category_id=ann["category_id"],
                                                   iscrowd=int(ann.get("iscrowd", 0))))
    seen = {row["sha"] for row in main_index["rows"]} | {row["sha"] for row in read(HERE/"omni_expanded/manifest.json")["rows"]}
    coco = read(HERE.parent / "local_policy/manifest.json")
    for row in coco["splits"]["cal_policy"]:
        seen.add(digest(HERE.parent / "local_policy/images" / row["dataset"] / row["file_name"]))
    rows, rejected = [], Counter()
    for iid in sorted(selected):
        row = frames[iid]
        output = ASSETS / "images" / row["patch"] / row["file_name"]
        if not output.is_file():
            rejected["missing_file"]+=1
            continue
        with Image.open(output) as image:
            if image.size != (row["width"], row["height"]):
                rejected["dimensions"]+=1
                continue
            image.verify()
        sha = digest(output)
        if sha in seen:
            rejected["duplicate_bytes"]+=1
            continue
        seen.add(sha)
        rows.append(dict(row,id=len(main_index["rows"])+len(rows),split="train",sha=sha,path=str(output),
                         known=list(main_index["categories"].values())))
    added = {a["category_id"] for row in rows for a in row["annotations"]}
    result = dict(rows=rows, main=digest(INDEX), plan=digest(RARE_PLAN), archives=source_receipts,
                  targeted=len(missing), added_classes=sorted(added & missing), remaining=sorted(missing-added),
                  annotated_train_classes=len(covered | added), rejected=dict(rejected))
    save_json(RARE_INDEX, result)
    print("rare prepared", {k:v for k,v in result.items() if k!="rows"}, "images", len(rows), flush=True)


if __name__=="__main__":
    main()
