"""Build stable native-query targets for a base-class pixel ownership head."""
from collections import Counter, defaultdict
import gzip
import json

import numpy as np
import torch
from torchvision.ops import box_iou

from dense_donor import ROOT,save_json,vocabulary
from counterfactual import gt_arrays
from operational_cache import sources

COLUMNS = ("image","query","annotation","target","iou","margin","actual_graph_pair","distinct_pair")


@torch.inference_mode()
def build(split,polygons):
    source,tag,_ = sources(split)
    manifest = json.loads((source/"manifest.json").read_text())
    names,nfine,_ = vocabulary(manifest,tag)
    assert nfine==48
    categories = {manifest["categories"][n] for n in names if n in manifest["categories"]}
    index = defaultdict(lambda:defaultdict(list))
    for real,prefix in ((True,"identity_pairs"),(False,"identity_enriched")):
        a = np.load(ROOT/f"{prefix}_{split}.npz")["rows"]
        for row in a[a[:,7]>0]:
            iid,q,r,state,s,t = map(int,row[[0,1,4,7,8,9]])
            index[iid][q].append((s,real,state==2))
            index[iid][r].append((t,real,state==2))
    records = []
    totals = Counter()
    for row in manifest["splits"][tag]:
        iid = row["id"]
        if iid not in index:
            continue
        annotations = [a for a in row["annotations"] if not a["iscrowd"] and a["category_id"] in categories]
        truth,labels = gt_arrays(row,names,manifest)
        assert len(annotations)==len(labels)
        pack = torch.load(source/"cache"/tag/f"{iid}.pt",map_location="cpu",weights_only=True)
        queries = sorted(index[iid])
        ious = box_iou(pack["boxes"][queries],truth)
        for j,q in enumerate(queries):
            candidates = index[iid][q]
            g = {s for s,_,_ in candidates}
            totals["query_pool"] += 1
            if len(g)!=1:
                totals["conflicting_targets"] += 1
                continue
            t = g.pop()
            own = float(ious[j,t])
            other = float(ious[j,torch.arange(len(truth))!=t].max()) if len(truth)>1 else 0.
            if own<.50 or own-other<.10:
                totals["ambiguous"] += 1
                continue
            annotation = annotations[t]["id"]
            assert str(annotation) in polygons
            real = any(f for _,f,_ in candidates)
            distinct = any(d for _,_,d in candidates)
            records.append((iid,q,annotation,t,own,own-other,int(real),int(distinct)))
    a = np.asarray(records)
    assert len(a)==len(np.unique(a[:,:2],axis=0))
    path = ROOT/f"mask_query_index_{split}.npz"
    with path.with_suffix(".part").open("wb") as out:
        np.savez_compressed(out,rows=a,columns=np.array(COLUMNS))
    path.with_suffix(".part").replace(path)
    return dict(**totals,queries=len(a),images=len(np.unique(a[:,0])),instances=len(np.unique(a[:,2])),
                actual_graph_queries=int(a[:,6].sum()),distinct_pair_queries=int(a[:,7].sum()))


if __name__=="__main__":
    torch.set_num_threads(2)
    with gzip.open(ROOT/"base_instance_polygons.json.gz","rt",encoding="utf-8") as f:
        masks = json.load(f)
    result = {}
    for split in ("train","mine","cal"):
        result[split] = build(split,masks[split])
        print(split,result[split],flush=True)
    save_json(ROOT/"mask_query_supply.json",result)
