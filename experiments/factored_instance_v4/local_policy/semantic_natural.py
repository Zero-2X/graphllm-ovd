"""Freeze the expanded same-box classifier and calibrate it on natural held-out images."""
import argparse
from collections import Counter
from itertools import groupby
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"visual_pair"))
from competition_cache import absolute_features,actual_donors
from counterfactual import ImageReplay,make_base
from dense_donor import ROOT,V1,save_json,vocabulary
from finish_gate import describe,denominators
from local_runtime import Runtime
from relabel import RelabelGate
from semantic_action_pool import eligible_actions
from semantic_evidence import FIELDS,baseline_evidence,load_patch_model,paired_inputs
from semantic_graph_refresh import NAME as ONLINE_NAME, PREFIX
from semantic_route_train import INPUT,read

SHARDS = 4


class SemanticRuntime:
    def __init__(self,names,nfine,parents):
        self.native = Runtime(names,nfine,parents)
        self.nfine = nfine
        self.patch = load_patch_model()
        qualification = json.loads((ROOT/f"{ONLINE_NAME}_qualification.json").read_text())
        assert qualification["passed"]
        assert json.loads((ROOT/f"{ONLINE_NAME}_result.json").read_text())["passed"]
        checkpoint = torch.load(ROOT/f"{ONLINE_NAME}_best.pt",map_location="cpu",weights_only=True)
        assert checkpoint["epoch"]==qualification["best_epoch"] and checkpoint["width"]==len(INPUT)
        self.model = RelabelGate(len(INPUT)).eval()
        self.model.load_state_dict(checkpoint["model"])

    @torch.inference_mode()
    def encode(self,image):
        runtime = self.native
        pack,field,scale = runtime.native_pack(image)
        base = make_base(pack,self.nfine)
        choices = eligible_actions(pack,base,self.nfine)
        if not choices:
            return pack,base,np.empty((0,4),dtype=np.int64),None,None,None
        keys = np.array(choices,dtype=np.int64)
        q,c,d,slot = keys.T
        old = base["label"][slot].numpy()
        roles = np.column_stack((old,c))
        edges = actual_donors(pack,self.nfine)
        donor = np.column_stack(([edges.get((j,o),-1) for j,o in zip(q,old)],d))
        graph,spatial = paired_inputs(runtime.reader,runtime.alignment,runtime.semantic,pack,q,roles)
        boxes,logits = pack["boxes"].cuda(),pack["logits"]
        f = absolute_features(runtime,self.patch,boxes,logits,field,scale,q,roles,donor,graph,spatial)
        corroboration,same = baseline_evidence(pack["boxes"][q],base,q,roles)
        x = np.column_stack((f[:,0],f[:,1]-f[:,0],corroboration,same)).astype(np.float32)
        assert x.shape==(len(q),len(INPUT))
        post = self.model.posterior(torch.from_numpy(x)).numpy()
        score = post[:,1]-post[:,2]
        return pack,base,keys,x,score,np.column_stack((corroboration[:,1]>0,same))


@torch.inference_mode()
def smoke():
    assert (HERE/"SEMANTIC_NATURAL_PLAN_ZH.md").exists()
    manifest = json.loads((V1/"manifest.json").read_text())
    names,nfine,parents = vocabulary(manifest,"cal")
    runtime = SemanticRuntime(names,nfine,parents)
    cached = read(("cal",),PREFIX)
    covered = np.unique(cached.rows[cached.rows[:,10]>0,0]).astype(int)
    chosen = set(covered[:3])
    candidates = set(cached.rows[:,0].astype(int))
    chosen.add(next(r["id"] for r in manifest["splits"]["cal"] if r["id"] in candidates-chosen
                    and any(a["iscrowd"] for a in r["annotations"])))
    trace = []
    for row in manifest["splits"]["cal"]:
        iid = row["id"]
        if iid not in chosen:
            continue
        image = Image.open(V1/"images"/row["dataset"]/row["file_name"]).convert("RGB")
        pack,base,keys,x,score,_ = runtime.encode(image)
        frozen = torch.load(V1/"cache/cal"/f"{iid}.pt",map_location="cpu",weights_only=True)
        for name in ("boxes","logits","ids","node_ids","i","j","valid","high","labels","grid","features","stats"):
            assert torch.equal(pack[name],frozen[name]),(iid,name)
        check = np.flatnonzero(cached.rows[:,0]==iid)
        lookup = {(q,c):j for j,(q,c,_,_) in enumerate(keys)}
        actual = np.array([lookup[tuple(k.astype(int))] for k in cached.rows[check][:,[1,3]]])
        assert np.array_equal(keys[actual][:,[2,3]],cached.rows[check][:,[4,5]])
        expect = cached.evidence[check][:,INPUT]
        p = runtime.model.posterior(torch.from_numpy(expect)).numpy()
        ref = p[:,1]-p[:,2]
        error = float(np.abs(x[actual]-expect).max())
        delta = float(np.abs(score[actual]-ref).max())
        winner_equal = ref.argmax()==score[actual].argmax()
        diff = np.abs(x[actual]-expect)
        j,k = np.unravel_index(diff.argmax(),diff.shape)
        trace.append(dict(image=iid,labeled_candidates=len(actual),online_candidates=len(keys),
                          max_feature_error=error,max_utility_error=delta,winner_equal=bool(winner_equal),
                          per_feature={FIELDS[t]:float(v) for t,v in zip(INPUT,diff.max(0))},
                          worst_query=keys[actual[j]].tolist(),worst_expected=float(expect[j,k]),worst_actual=float(x[actual[j],k])))
        print(trace[-1],flush=True)
    assert len(trace)==4
    passed = all(t["max_feature_error"]<=.01 and t["max_utility_error"]<=.002 and t["winner_equal"] for t in trace)
    save_json(HERE/"semantic_natural_smoke_diagnostic.json",dict(passed=passed,images=trace))
    assert passed
    save_json(HERE/"semantic_natural_smoke.json",dict(passed=True,model=ONLINE_NAME,images=trace))


@torch.inference_mode()
def run(shard):
    assert json.loads((HERE/"semantic_natural_fetch_receipt.json").read_text())["hash_disjoint"]
    assert json.loads((HERE/"semantic_natural_smoke.json").read_text())["passed"]
    manifest = json.loads((HERE/"manifest.json").read_text())
    names,nfine,parents = vocabulary(manifest,"cal_policy")
    runtime = SemanticRuntime(names,nfine,parents)
    result = []
    rows = manifest["splits"]["cal_policy"][shard::SHARDS]
    for j,row in enumerate(rows,1):
        iid = row["id"]
        im = Image.open(HERE/"images"/row["dataset"]/row["file_name"]).convert("RGB")
        pack,base,keys,x,score,coverage = runtime.encode(im)
        item = dict(image=iid,candidates=len(keys))
        replay = ImageReplay(row,pack,base,manifest,names)
        actions = [("replace",int(q),int(c),int(slot)) for q,c,_,slot in keys]
        old_true,correct = denominators(replay,[(a,float(base["score"][a[3]])) for a in actions])
        item.update(old_true=old_true,correct_receivers=correct)
        if len(keys):
            order = np.lexsort((keys[:,1],keys[:,0],-score))
            ix = order[0]
            if score[ix]>0:
                q,c,d,slot = map(int,keys[ix])
                outcome = describe(replay,actions[ix],float(base["score"][slot]))
                outcome["safe_fix"] = int(outcome["semantic_fix"] and not outcome["semantic_bad"] and not outcome["lost_floor"])
                other,same = map(bool,coverage[ix])
                outcome["covered_fix"] = outcome["safe_fix"]*int(other or same)
                outcome["other_native_fix"] = outcome["safe_fix"]*int(other)
                item.update(utility=float(score[ix]),query=q,proposed=c,donor=d,covered=other or same,
                            other_native=other,same_query=same,outcome=outcome)
        result.append(item)
        if j%25==0 or j in (1,len(rows)):
            print("semantic natural",shard,j,len(rows),flush=True)
    save_json(HERE/f"semantic_natural_{shard}.json",result)


def totals(rows):
    outcome = Counter()
    for item in rows:
        outcome.update(item["outcome"])
    outcome["edits"] = len(rows)
    return dict(outcome)


def calibrate():
    items = []
    for i in range(SHARDS):
        items.extend(json.loads((HERE/f"semantic_natural_{i}.json").read_text()))
    manifest = json.loads((HERE/"manifest.json").read_text())
    assert len(items)==len({a["image"] for a in items})==1320
    assert {a["image"] for a in items}=={r["id"] for r in manifest["splits"]["cal_policy"]}
    order = sorted((r for r in items if "utility" in r),key=lambda r:-r["utility"])
    accumulated = Counter()
    best = None
    history = []
    for threshold,group in groupby(order,key=lambda a:a["utility"]):
        batch = list(group)
        accumulated.update(totals(batch))
        n = accumulated["edits"]
        allowed = (accumulated["safe_fix"]>=5 and accumulated["other_native_fix"]>=1 and
                   accumulated["safe_fix"]/n>=.80 and not (accumulated["semantic_bad"] or
                   accumulated["lost_floor"] or accumulated["fp_delta_035"]))
        point = dict(accumulated,threshold=threshold,qualified=allowed)
        history.append(point)
        if allowed and (best is None or (point["safe_fix"],-n)>(best["safe_fix"],-best["edits"])):
            best = point
    selected = [r for r in order if best is not None and r["utility"]>=best["threshold"]]
    result = dict(images=len(items),candidates=sum(r["candidates"] for r in items),
                  old_true=sum(r["old_true"] for r in items),correct_receivers=sum(r["correct_receivers"] for r in items),
                  best=best,passed=best is not None,all_winners=totals(order),selected=totals(selected),
                  covered=totals([r for r in selected if r["covered"]]),
                  uncovered=totals([r for r in selected if not r["covered"]]),
                  thresholds=history,eval_opened=False)
    save_json(HERE/"semantic_natural_result.json",result)
    print({k:v for k,v in result.items() if k!="thresholds"},flush=True)


if __name__=="__main__":
    p = argparse.ArgumentParser()
    modes = p.add_mutually_exclusive_group(required=True)
    modes.add_argument("--smoke",action="store_true")
    modes.add_argument("--calibrate",action="store_true")
    modes.add_argument("--shard",type=int,choices=range(SHARDS))
    a = p.parse_args()
    if a.smoke:
        smoke()
    elif a.calibrate:
        calibrate()
    else:
        run(a.shard)
