"""Fit the sole trainable three-outcome head and test optimization convergence."""
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from selective import AdoptionHead,FrozenStar,V1,prepare_image,training_targets,vocabulary

SEED=20261110
EPOCHS=24
LR=.001
BATCH=512


def save_json(file,data):
    path=Path(file)
    tmp=path.with_suffix(".part")
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False))
    tmp.replace(path)


def load_examples(manifest,split,encoder):
    path=Path("examples_"+split+".pt")
    if path.is_file():
        data=torch.load(path,map_location="cpu",weights_only=True)
    else:
        names,nfine,_=vocabulary(manifest,split)
        encoder.set_vocabulary(names,nfine)
        examples,old,y,ids=[],[],[],[]
        for i,row in enumerate(manifest["splits"][split],1):
            base,prop,x=prepare_image(row,manifest,names,nfine,encoder,V1/"cache"/split/(str(row["id"])+".pt"))
            if prop:
                labels,mask=training_targets(row,manifest,names,base,prop)
                if mask.any():
                    examples.append(x[mask])
                    old.append(prop["old"][mask])
                    y.append(labels[mask])
                    ids.append(torch.full((int(mask.sum()),),row["id"],dtype=torch.long))
            if i%200==0:
                print("examples",split,i,flush=True)
        data=dict(x=torch.cat(examples),old=torch.cat(old),y=torch.cat(y),image=torch.cat(ids))
        torch.save(data,path)
    counts=torch.bincount(data["y"],minlength=3).tolist()
    print("selected",split,"images",len(data["image"].unique()),"samples",len(data["y"]),"neither/gain/harm",counts,flush=True)
    return data


@torch.inference_mode()
def validate(head,data):
    head.eval()
    losses=[]
    accuracy=0
    for ix in torch.arange(len(data["y"])).split(BATCH):
        target=data["y"][ix].cuda()
        logits=head(data["x"][ix].cuda(),data["old"][ix].cuda())
        losses.append(F.cross_entropy(logits,target,reduction="none").cpu())
        accuracy+=int((logits.argmax(-1)==target).sum())
    per=torch.cat(losses).numpy()
    return float(per.mean()),per,accuracy/len(per)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_float32_matmul_precision("high")
    manifest=json.loads((V1/"manifest.json").read_text())
    encoder=FrozenStar() if not all(Path("examples_"+p+".pt").is_file() for p in ("train","cal")) else None
    train=load_examples(manifest,"train",encoder)
    cal=load_examples(manifest,"cal",encoder)
    del encoder
    torch.cuda.empty_cache()
    head=AdoptionHead(train["x"].shape[1]).cuda()
    opt=torch.optim.AdamW(head.parameters(),lr=LR,weight_decay=.01)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS,eta_min=LR*.025)
    initial,initial_per,_=validate(head,cal)
    best=float("inf")
    history=[]
    for epoch in range(1,EPOCHS+1):
        head.train()
        total=0
        for ix in torch.randperm(len(train["y"])).split(BATCH):
            opt.zero_grad(set_to_none=True)
            z=head(train["x"][ix].cuda(),train["old"][ix].cuda())
            loss=F.cross_entropy(z,train["y"][ix].cuda())
            assert torch.isfinite(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(),1.0,error_if_nonfinite=True)
            opt.step()
            total+=float(loss)*len(ix)
        sched.step()
        val,per,acc=validate(head,cal)
        assert np.isfinite(val) and np.isfinite(per).all()
        rec=dict(epoch=epoch,train=total/len(train["y"]),cal=val,accuracy=acc,lr=opt.param_groups[0]["lr"])
        history.append(rec)
        if val<best:
            best=val
            torch.save(dict(state=head.state_dict(),dim=train["x"].shape[1],epoch=epoch),"best.pt")
        save_json("training.json",history)
        print("epoch",rec,flush=True)
    torch.save(dict(state=head.state_dict(),dim=train["x"].shape[1],epoch=EPOCHS),"final.pt")
    t=np.array([r["train"] for r in history[-5:]])
    v=np.array([r["cal"] for r in history[-5:]])
    plateau=abs(float(t[:2].mean()-t[-2:].mean()))/float(t[:2].mean())
    slope=abs(float(np.polyfit(np.arange(5),v,1)[0]))
    _,group=np.unique(cal["image"].numpy(),return_inverse=True)
    n=int(group.max())+1
    totals=np.bincount(group,weights=per-initial_per)
    counts=np.bincount(group)
    boot=np.random.default_rng(SEED).integers(0,n,size=(1000,n))
    se=float((totals[boot].sum(1)/counts[boot].sum(1)).std(ddof=1))
    tol=max(.005,2*se)
    checks=dict(schedule=abs(history[-1]["lr"]-LR*.025)<1e-9,plateau=plateau<=.025,
                cal_slope=slope<=tol/4,cal_spread=float(v.max()-v.min())<=tol,cal_gap=float(v[-1]-best)<=tol)
    result=dict(passed=all(checks.values()),checks=checks,initial_cal=initial,best_cal=best,
                final_cal=float(v[-1]),plateau=plateau,cal_slope=slope,cal_tolerance=tol,
                train_counts=torch.bincount(train["y"],minlength=3).tolist(),
                cal_counts=torch.bincount(cal["y"],minlength=3).tolist(),epochs=EPOCHS)
    save_json("qualification.json",result)
    print("qualification",result,flush=True)


if __name__=="__main__":
    main()
