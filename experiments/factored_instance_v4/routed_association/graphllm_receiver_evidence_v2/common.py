"""Versioned paths, fixed inputs, and provenance for the receiver evidence run."""
import hashlib
import json
from pathlib import Path

from graphllm_local_message_unique_v2.bridge import (
    BASE, FOLDER, INDEX, KEEP, RARE_INDEX, cache_write, data_sources, digest, load, read, safe_frame, save_json,
)

CODE = Path(__file__).resolve().parent
OUT = FOLDER/CODE.name
CONFIG = CODE/"config.json"
HISTORY = FOLDER/"graphllm_local_message_unique_v2"
OLD_CODE = CODE.parent/HISTORY.name
SCHEMA = "receiver_evidence_v2/canonical/native16/local9/edge12/message128/gain147"


def signature(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",",":")).encode()).hexdigest()


def config():
    cfg = read(CONFIG)
    assert cfg["version"]==CODE.name and cfg["message"]["js_affects_decision"] is False
    return cfg


def vocabulary(doc):
    names=doc["base"]
    categories=[doc["categories"][name] for name in names]
    assert len(names)==len(set(names))==len(set(categories))==config()["classes"]
    official={x["id"]:x["name"] for x in doc["official_categories"]}
    assert set(official)==set(categories) and doc["names"][:len(names)]==names
    return dict(text_to_canonical=categories,canonical_to_text=dict(zip(categories,names)),
                official_names=official,aliases={official[c]:name for c,name in zip(categories,names)},
                shared_coco=doc.get("shared_coco",{}))


def native_sources():
    """Check inherited frozen weight/preprocessing identities without opening holdout GT."""
    frozen=read(HISTORY/"resolved_config.json")
    exclude={str((FOLDER/"graphllm_local_message_v1"/"manifest.json").resolve())}
    entries={name:sha for name,sha in frozen["source_hashes"].items() if name not in exclude}
    for name,sha in entries.items():
        path=Path(name)
        stat=path.stat()
        if [stat.st_size,stat.st_mtime_ns]!=frozen["source_stamps"][name]:
            assert digest(path)==sha,name
    for name,sha in frozen["code_hashes"].items():
        assert digest(OLD_CODE/name)==sha,name
    return dict(resolved=digest(HISTORY/"resolved_config.json"),sources=entries,code=frozen["code_hashes"])


def data_lock():
    return read(OUT/"data_lock.json")


def feature_lock():
    lock=data_lock()
    for name,sha in lock["feature_code"].items():
        assert digest(CODE/name)==sha,name
    assert digest(CONFIG)==lock["config_hash"]
    return lock["feature_lock"]


def checked(stage):
    b=read(OUT/"BUILD_COMPLETE.json")
    assert b["feature_lock"]==feature_lock()
    assert b["data"]==digest(OUT/"data_lock.json") and b["native_stats"]==digest(OUT/"native_stats.pt")
    assert b["smoke"]==digest(OUT/"smoke.json")
    for name,sha in b["code"].items():
        assert digest(CODE/name)==sha,name
    assert native_sources()==data_lock()["prior"]
    return b


def model_lock(stage):
    build=checked(stage)
    chain={"build":digest(OUT/"BUILD_COMPLETE.json"),"stage":stage}
    for previous in ("self","message","gain"):
        if stage==previous:
            break
        chain[previous]=digest(OUT/f"{previous}_best.pt")
        rec=read(OUT/f"{previous}_qualification.json")
        assert rec["completed"] and rec["best_hash"]==chain[previous]
    return chain


def locked_best(stage):
    rec=read(OUT/f"{stage}_qualification.json")
    path=OUT/f"{stage}_best.pt"
    assert rec["completed"] and rec["best_hash"]==digest(path)
    state=load(path)
    assert state["lock"]==model_lock(stage) and state["point"]==rec["best_point"]
    return state


def annotated_split(split):
    doc,splits=data_sources()
    assert split in ("train","monitor","policy","eval")
    expected=data_lock()["sets"][split]
    assert signature([safe_frame(r) for r in splits[split]])==expected["frames"]
    return doc,splits[split]


def blind_document(doc,rows):
    return {**{k:doc[k] for k in ("names","base","categories","parents","official_categories","shared_coco")},
            "rows":[safe_frame(r) for r in rows]}


def new_inputs(doc,split):
    result=read(OUT/f"images_{split}.json")
    assert digest(OUT/f"images_{split}.json")==data_lock()["sets"][split]["blind"]
    assert set(result)=={"names","base","categories","parents","official_categories","shared_coco","rows"}
    assert result["names"]==doc["names"] and result["categories"]==doc["categories"]
    assert all(set(r)=={"id","source_id","file_name","path","width","height","sha"} for r in result["rows"])
    return result


def init_data():
    cfg=config()
    prior=native_sources()
    doc,splits=data_sources()
    holdout=read(OUT/"manifest.json")
    assert len(holdout["rows"])==cfg["evaluation"]["holdout_count"]
    assert vocabulary(doc)==vocabulary(holdout)
    historical=read(HISTORY/"holdout_images.json")
    populations=dict(splits,historical=historical["rows"],holdout=holdout["rows"])
    for k in ("source_id","file_name","sha"):
        seen=set()
        for name,rows in populations.items():
            current={r[k] for r in rows}
            assert len(current)==len(rows) and not current&seen,(name,k)
            seen|=current
    for name,rows in populations.items():
        assert len({r["id"] for r in rows})==len(rows),name
    assert not ({r["id"] for r in splits["train"]}&{r["id"] for r in splits["monitor"]+splits["policy"]+splits["eval"]})
    sets={}
    for name,rows in populations.items():
        file=OUT/f"images_{name}.json"
        save_json(file,blind_document(doc,rows))
        sets[name]=dict(n=len(rows),frames=signature([safe_frame(r) for r in rows]),blind=digest(file))
    historical_policy=read(HISTORY/"locked_policy.json")
    frozen_v1={name:digest(HISTORY/name) for name in ("semantic_best.pt","gain_best.pt","locked_policy.json","predictions_manifest.json","result.json")}
    assert historical_policy["semantic"]==frozen_v1["semantic_best.pt"] and historical_policy["gain"]==frozen_v1["gain_best.pt"]
    assert read(HISTORY/"exploratory_audit.json")["counts"]["new_full"]["executed"]==71
    features={name:digest(CODE/name) for name in ("native_bridge.py","targets.py","data.py","inputs.py")}
    key=dict(prior=prior,config_hash=digest(CONFIG),sets=sets,vocabulary=vocabulary(doc),
             index=digest(INDEX),rare=digest(RARE_INDEX),annotations=digest(OUT/"manifest.json"),
             holdout_sampler=digest(CODE/"prepare_holdout.py"),feature_code=features,frozen_v1=frozen_v1)
    key.update(feature_lock=signature(key))
    p=OUT/"data_lock.json"
    if p.exists():
        prior=read(p)
        if prior==json.loads(json.dumps(key)):
            return prior
        assert prior["sets"]==sets and prior["annotations"]==key["annotations"]
        assert not (OUT/"smoke.json").exists() and not any(OUT.glob("visual_*.pt"))
    save_json(p,key)
    return key
