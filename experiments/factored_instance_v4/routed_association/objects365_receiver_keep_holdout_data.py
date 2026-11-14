"""Select fresh Objects365 images from the same archive as the original cohort."""
from collections import Counter, defaultdict, deque
import hashlib
from pathlib import Path
import re
import tarfile

from PIL import Image

from objects365_data import ASSETS, BASE, FOLDER, HERE, INDEX, annotation_file, digest, read, records, save_json
from objects365_receiver_keep_gate import OUT as KEEP, qualified
from objects365_rare import RARE_INDEX


OUT = FOLDER/"receiver_keep_holdout"
PLAN = HERE/"OBJECTS365_RECEIVER_KEEP_HOLDOUT.md"
NEW = OUT/"manifest.json"
COUNT = 1024


def exclusions():
    names,ids,sha = set(),set(),set()
    docs = sorted((BASE/"experiments").rglob("manifest.json"))+[RARE_INDEX]
    for path in docs:
        data = read(path)
        todo = [data]
        while todo:
            row = todo.pop()
            if isinstance(row,list):
                todo.extend(row)
            elif isinstance(row,dict):
                todo.extend(row.values())
                path_value = row.get("file_name",row.get("path",""))
                name = Path(path_value).name if isinstance(path_value,str) else ""
                if name.startswith("objects365_"):
                    names.add(name)
                    if "source_id" in row:
                        ids.add(row["source_id"])
                if isinstance(row.get("sha"),str) and len(row["sha"])==64:
                    sha.add(row["sha"])
    for path in (BASE/"data/omnilabel/annotations").glob("dataset_all_*.json"):
        names.update(Path(r["file_name"]).name for r in read(path)["images"]
                     if r["file_name"].startswith("object365/"))
    coco = read(HERE.parent/"local_policy"/"manifest.json")
    for row in coco["splits"]["cal_policy"]:
        sha.add(digest(HERE.parent/"local_policy"/"images"/row["dataset"]/row["file_name"]))
    return names,ids,sha,len(docs)


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    doc = read(INDEX)
    _,risk = qualified()
    selected = read(KEEP/"policy.json")["points"]["mixed_entropy"]["limited"]["point"]
    assert selected["cutoff"] is not None and selected["correct"]>0
    locked = dict(index=digest(INDEX),rare=digest(RARE_INDEX),plan=digest(PLAN),
                  policy=digest(KEEP/"policy.json"),qualification=digest(KEEP/"qualification.json"),
                  checkpoint=risk["best_hash"])
    if NEW.exists():
        saved = read(NEW)
        assert saved["lock"]==locked and len(saved["rows"])==COUNT
        print("existing fresh cohort",saved["counts"],flush=True)
        return
    patches = [name for name in doc["archives"] if re.fullmatch(r"patch\d+\.tar\.gz",name)]
    assert len(patches)==1
    archive_path = ASSETS/patches[0]
    excluded,used_ids,seen,n_manifests = exclusions()
    with tarfile.open(archive_path,"r|gz") as archive:
        candidates = {Path(m.name).name for m in archive if m.isfile() and m.name.lower().endswith(".jpg")}
    candidates -= excluded
    annotation = annotation_file()
    frames = {}
    for r in records(annotation,"images"):
        name = Path(r["file_name"]).name
        if name in candidates and r["id"] not in used_ids:
            frames[r["id"]] = dict(source_id=r["id"],file_name=name,width=r["width"],height=r["height"],annotations=[])
    for r in records(annotation,"annotations"):
        if r["image_id"] in frames and min(r["bbox"][2:])>0:
            frames[r["image_id"]]["annotations"].append(dict(id=r["id"],bbox=r["bbox"],
                category_id=r["category_id"],iscrowd=int(r.get("iscrowd",0))))
    print("unused frames with annotations scanned",len(frames),flush=True)
    repeats,frequency = {},Counter()
    for iid,row in frames.items():
        n = Counter(r["category_id"] for r in row["annotations"]
                    if not r["iscrowd"] and r["bbox"][2]*r["bbox"][3]>=196)
        repeats[iid] = {c for c,count in n.items() if count>=2}
        frequency.update(repeats[iid])
    order = sorted(frames,key=lambda i:hashlib.sha256(f"receiver-keep-new-20261112/{i}".encode()).digest())
    buckets = defaultdict(deque)
    for iid in order:
        if repeats[iid]:
            owner = min(repeats[iid],key=lambda c:(frequency[c],c))
            buckets[owner].append(iid)
    selected = []
    while buckets and len(selected)<COUNT+128:
        for c in sorted(buckets):
            selected.append(buckets[c].popleft())
            if not buckets[c]:
                del buckets[c]
            if len(selected)==COUNT+128:
                break
    assert len(selected)>=COUNT
    names = {frames[i]["file_name"] for i in selected}
    dst = OUT/"images"
    dst.mkdir(exist_ok=True)
    with tarfile.open(archive_path,"r|gz") as archive:
        for m in archive:
            name = Path(m.name).name
            if m.isfile() and name in names:
                path = dst/name
                if not path.exists():
                    temp = path.with_suffix(".part")
                    with archive.extractfile(m) as src:
                        temp.write_bytes(src.read())
                    temp.replace(path)
    rows,rejected,classes = [],Counter(),set()
    for iid in selected:
        r = frames[iid]
        path = dst/r["file_name"]
        with Image.open(path) as image:
            if image.size!=(r["width"],r["height"]):
                rejected.update(dimensions=1)
                continue
            image.verify()
        sha = digest(path)
        if sha in seen:
            rejected.update(duplicate_sha256=1)
            continue
        assert r["file_name"] not in excluded and iid not in used_ids
        seen.add(sha)
        classes.update(a["category_id"] for a in r["annotations"])
        rows.append(dict(r,id=len(rows),path=str(path),sha=sha,split="holdout",
                         known=list(doc["categories"].values())))
        if len(rows)==COUNT:
            break
    assert len(rows)==COUNT and len({r["file_name"] for r in rows})==COUNT
    stats = dict(images=len(rows),classes=len(classes),rejected=dict(rejected),
                 excluded_names=len(excluded),manifests=n_manifests)
    # Keep the fixed vocabulary and the same 365-way prompt.
    result = {k:doc[k] for k in ("names","base","categories","parents","official_categories","shared_coco")}
    save_json(NEW,dict(result,rows=rows,counts=stats,lock=locked,
                       archive=archive_path.name,archive_sha256=doc["archives"][archive_path.name]))
    print("fresh cohort",stats,flush=True)


if __name__=="__main__":
    prepare()

