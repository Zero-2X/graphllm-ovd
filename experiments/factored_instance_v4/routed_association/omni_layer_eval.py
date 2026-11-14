"""Lock support and mixed entropy on OmniLabel, then verify on COCO."""
import numpy as np
import torch

from graph_residual_eval import check_action, detector_metrics, frontier, passed, point_summary, route_outcomes, route_setup
from layer_entropy_cache import FOLDER as COCO_FEATURES, SHARDS as COCO_SHARDS
from layer_entropy_eval import entropy
from omni_layer_cache import FOLDER, HERE, INDEX, digest, read
from omni_layer_train import load_blocks, qualified
from ownership_graph_audit import auc
from route_screen import save_json


COCO_PLAN=HERE/"OMNI_LAYER_COCO_CAL_ZH.md"


def score(records,model):
    x=torch.cat([r["x"] for r in records]).float().cuda()
    assert x.shape[1:]==(3,3584) and len(x)==len(records)
    pred=model(x.flatten(0,1)).softmax(1)[:,1].reshape(-1,3)
    mean=pred.mean(1)
    avg=entropy(pred).mean(1)
    js=(entropy(mean)-avg).clamp_min(0)
    output=[]
    for row,p,avg_p,spread in zip(records,pred.tolist(),mean.tolist(),js.tolist()):
        q,c,slot=row["actions"][0].tolist()
        output.append(dict(image=row["image"],query=q,proposed=c,slot=slot,score=avg_p,views=p,js=spread))
    return output


def options(rows, cap):
    return [r for r in rows if r["score"]>.5 and (cap is None or r["js"]<=cap)]


def discriminate(rows):
    is_fix=np.array([r["correct"] for r in rows])
    return dict(rows=len(rows),positives=int(is_fix.sum()),mean_auc=auc(np.array([r["score"] for r in rows]),is_fix),
                low_js_auc=auc(-np.array([r["js"] for r in rows]),is_fix))


def coco_calibration(model):
    rows=[]
    for shard in range(COCO_SHARDS):
        cache=torch.load(COCO_FEATURES/f"route_{shard}.pt",map_location="cpu",weights_only=True)
        assert not cache["eval_opened"] and all("outcomes" not in r for r in cache["records"])
        rows.extend(cache["records"])
    by_image={r["image"]:r for r in score(rows,model)}
    labels=route_outcomes(by_image)
    train_sha={r["sha"] for r in read(INDEX)["rows"]}
    manifest,_,_,_=route_setup()
    duplicate={r["id"] for r in manifest["splits"]["route_train"]
               if digest(HERE/"images"/r["dataset"]/r["file_name"]) in train_sha}
    labels=[r for r in labels if r["image"] not in duplicate]
    good=[r["js"] for r in labels if r["target"]==1. and r["views"][0]>.5]
    cap=float(np.percentile(good,90)) if good else 0.
    limits=dict(coco_cal_mean=None,coco_cal_mixed=cap)
    thresholds={k:frontier(options(labels,v),[],(0,),True)[0] for k,v in limits.items()}
    return dict(plan=digest(COCO_PLAN),duplicates=len(duplicate),thresholds=thresholds,cap=cap,reference=len(good)),limits


@torch.inference_mode()
def main():
    model,info=qualified()
    data,_=load_blocks()
    predictions=score(data["policy"],model)
    policy=[dict(p,**r["outcomes"][0]) for p,r in zip(predictions,data["policy"])]
    observed=[r for r in policy if r["target"]>=0]
    good=[r["js"] for r in observed if r["correct"] and r["views"][0]>.5]
    cap=float(np.percentile(good,90)) if good else 0.
    choices=dict(mean=None,mixed_entropy=cap)
    thresholds={name:frontier(options(observed,limit),[],(0,),True)[0] for name,limit in choices.items()}
    unknown={k:sum(r["target"]<0 and passed(r["score"],thresholds[k]["threshold"]) for r in options(policy,v)) for k,v in choices.items()}
    policy_lock=dict(receipt=info["receipt"],index=digest(INDEX),thresholds=thresholds,unknown_above_threshold=unknown,
                     labeled=len(observed),candidates=len(policy),cap=cap,cap_reference_count=len(good),coco_outcomes_read=False)
    save_json(FOLDER/"policy.json",policy_lock)
    print("OmniLabel policy locked",policy_lock,flush=True)
    coco_policy,coco_limits=coco_calibration(model)
    save_json(FOLDER/"coco_policy.json",dict(coco_policy,training=info["receipt"],held_outcomes_read=False))
    print("COCO validation policy locked",coco_policy,flush=True)
    choices.update(coco_limits)
    all_thresholds={**thresholds,**coco_policy["thresholds"]}

    coco=[]
    for shard in range(COCO_SHARDS):
        doc=torch.load(COCO_FEATURES/f"natural_{shard}.pt",map_location="cpu",weights_only=True)
        assert not doc["eval_opened"] and all("outcomes" not in r for r in doc["records"])
        coco.extend(doc["records"])
    scores=score(coco,model)
    save_json(FOLDER/"coco_scores.json",dict(policy=digest(FOLDER/"policy.json"),rows=scores,labels_read=False))
    by_image={r["image"]:r for r in scores}
    old=read(HERE/"receiver_admission_v3_result.json")
    energy=read(HERE/"energy_admission_result.json")
    foreground=read(HERE/"foreground_full_lock.json")
    cut=read(HERE/"foreground_admission_full_result.json")["foreground_calibrated"][0]["threshold"]
    anchors={r["image"] for r in old["anchors"]} | {r["image"] for r in foreground["natural"] if r["score"]>=cut}
    anchors |= set(energy["energy"]["ids"])
    rows=[]
    for prior in old["anchors"]+old["eligible"]:
        result=by_image[prior["image"]]
        check_action(result,prior)
        rows.append(dict(prior,score=result["score"],js=result["js"]))
    assert len(rows)==len(scores)==945 and sum(r["correct"] for r in rows)==157
    selected={k:[r for r in options(rows,v) if passed(r["score"],all_thresholds[k]["threshold"])] for k,v in choices.items()}
    verified_coco=[r for r in rows if r["correct"] or r["wrong"] or any(r["lost"])]
    result=dict(policy=policy_lock,coco_policy=coco_policy,omni_diagnostic=discriminate(observed),coco_diagnostic=discriminate(verified_coco),
                coco={k:point_summary(v,anchors) for k,v in selected.items()},
                frontier={k:frontier(options(rows,v),[],(0,1,2),True) for k,v in choices.items()},
                ap=detector_metrics(selected),coco_baseline=energy["energy"]["total"],eval_opened=False)
    save_json(FOLDER/"result.json",result)
    print("OmniLabel to COCO",result,flush=True)


if __name__ == "__main__":
    main()
