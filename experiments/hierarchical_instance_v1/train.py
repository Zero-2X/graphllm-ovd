"""Train one graph adapter. Efficacy evaluation is gated on calibration convergence."""
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torchvision.ops import box_iou

from cache import gt_arrays, vocabulary
from model import StarAdapter, per_row_loss, receiver_batch

EPOCHS=24
BATCH=64
LR=.001
SEED=20261110
WINDOW=5
PLATEAU=.025


def save_json(path,data):
    path=Path(path)
    temp=path.with_suffix(".part")
    temp.write_text(json.dumps(data,indent=2,allow_nan=False))
    temp.replace(path)


def load_split(manifest,split,names,parents):
    blocks=[]
    summary=Counter()
    admitted=[]
    parents=torch.tensor(parents,dtype=torch.long)
    for r in manifest["splits"][split]:
        pack=torch.load(Path("cache")/split/(str(r["id"])+".pt"),map_location="cpu",weights_only=True)
        screen=pack.get("screen",{})
        summary.update(screen)
        for k in ("leaf_opportunity","parent_opportunity","loc_opportunity"):
            summary[k+"_images"]+=int(screen.get(k,0)>0)
        rows=pack["train_ids"]
        if split=="train" and r["stratum"]!="random" and not any(
                screen.get(k,0) for k in ("leaf_opportunity","parent_opportunity","loc_opportunity")):
            continue
        if not len(rows):
            continue
        gt,classes=gt_arrays(r,names,manifest)
        batch=receiver_batch(pack,rows)
        match=box_iou(batch["box"],gt).argmax(-1)
        batch.update(y=classes[match],gt=gt[match],parent=parents[classes[match]],
                     image=torch.full((len(rows),),r["id"],dtype=torch.long))
        blocks.append(batch)
        admitted.append(r["id"])
    data={k:torch.cat([b[k] for b in blocks]) for k in blocks[0]}
    return data,dict(summary,images=len(manifest["splits"][split]),admitted=len(admitted),
                     rows=len(data["y"]),admitted_ids=admitted)


def iterate(data,order):
    for idx in order.split(BATCH):
        yield {k:v[idx].cuda(non_blocking=True) for k,v in data.items()}


@torch.inference_mode()
def validate(model,data,nfine,disabled=False):
    model.eval()
    losses=[]
    base=[]
    for batch in iterate(data,torch.arange(len(data["y"]))):
        logits,boxes=model(batch,disabled=disabled)
        losses.append(per_row_loss(batch,logits,boxes,nfine).cpu())
        base.append(per_row_loss(batch,batch["base"],batch["box"],nfine).cpu())
    loss=torch.cat(losses)
    return float(loss.mean()),(loss-torch.cat(base)).numpy()


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_float32_matmul_precision("high")
    manifest=json.loads(Path("manifest.json").read_text())
    names,nfine,parents=vocabulary(manifest,"train")
    train,train_stats=load_split(manifest,"train",names,parents)
    cal,cal_stats=load_split(manifest,"cal",names,parents)
    save_json("screen.json",{"train":train_stats,"cal":cal_stats})
    print("Screen", {k:{x:y for x,y in v.items() if x!="admitted_ids"}
                     for k,v in (("train",train_stats),("cal",cal_stats))},flush=True)
    model=StarAdapter().cuda()
    model.set_vocabulary(names,nfine,"cuda")
    sample=next(iterate(cal,torch.arange(min(BATCH,len(cal["y"])))))
    with torch.no_grad():
        untouched=model(sample)
        assert torch.equal(untouched[0],sample["base"]) and torch.equal(untouched[1],sample["box"])
        no_graph=dict(sample,valid=torch.zeros_like(sample["valid"]),loc=torch.zeros_like(sample["loc"]))
        identity=model(no_graph)
        assert torch.equal(identity[0],sample["base"]) and torch.equal(identity[1],sample["box"])
    trainable=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(trainable,lr=LR,weight_decay=.01)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS,eta_min=LR*.025)
    baseline,_=validate(model,cal,nfine,disabled=True)
    loss_history=[]
    cal_history=[]
    best=float("inf")
    differences=[]
    for epoch in range(1,EPOCHS+1):
        model.train()
        total=0.
        for batch in iterate(train,torch.randperm(len(train["y"]))):
            opt.zero_grad(set_to_none=True)
            logits,boxes=model(batch)
            objective=per_row_loss(batch,logits,boxes,nfine).mean()
            objective=objective+.01*(logits-batch["base"]).square().mean()
            assert torch.isfinite(objective),f"non-finite loss epoch {epoch}"
            objective.backward()
            gradient=torch.nn.utils.clip_grad_norm_(trainable,1.0,error_if_nonfinite=True)
            opt.step()
            total+=objective.item()*len(batch["y"])
        schedule.step()
        val,diff=validate(model,cal,nfine)
        assert np.isfinite(val) and np.isfinite(diff).all()
        loss_history.append(total/len(train["y"]))
        cal_history.append(val)
        differences.append(diff)
        print(f"epoch {epoch}/{EPOCHS} train {loss_history[-1]:.6f} cal {val:.6f} baseline {baseline:.6f} lr {opt.param_groups[0]['lr']:.8f}",flush=True)
        if val<best:
            best=val
            torch.save(model.adapter_state(),"best.pt")
        save_json("history.json",dict(train=loss_history,cal=cal_history,baseline=baseline,best=best))
    torch.save(model.adapter_state(),"final.pt")
    last=loss_history[-WINDOW:]
    plateau=abs(float(np.mean(last[:2])-np.mean(last[-2:])))/max(float(np.mean(last[:2])),1e-8)
    vals=np.array(cal_history[-WINDOW:])
    slope=abs(float(np.polyfit(np.arange(WINDOW),vals,1)[0]))
    _,group=np.unique(cal["image"].numpy(),return_inverse=True)
    n=int(group.max())+1
    weights=np.bincount(group)
    sums=np.bincount(group,weights=differences[-1])
    rng=np.random.default_rng(SEED)
    draw=rng.integers(0,n,size=(1000,n))
    stderr=float((sums[draw].sum(1)/weights[draw].sum(1)).std(ddof=1))
    uncertainty=max(.002,2*stderr)
    check=dict(schedule_complete=abs(opt.param_groups[0]["lr"]-LR*.025)<1e-9,
               plateau=plateau<=PLATEAU,
               cal_slope=slope<=uncertainty/(WINDOW-1),
               cal_spread=float(vals.max()-vals.min())<=uncertainty,
               cal_final_gap=float(vals[-1]-best)<=uncertainty)
    result=dict(passed=all(check.values()),checks=check,plateau=plateau,cal_slope=slope,
                cal_spread=float(vals.max()-vals.min()),cal_gap=float(vals[-1]-best),
                paired_image_se=stderr,cal_tolerance=uncertainty,baseline=baseline,
                cal_best=best,cal_final=float(vals[-1]),epochs=EPOCHS)
    save_json("qualification.json",result)
    print("Qualification",result,flush=True)


if __name__=="__main__":
    main()
