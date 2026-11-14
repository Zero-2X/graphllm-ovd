"""The three prescribed optimization stages and their immutable best snapshots."""
from contextlib import nullcontext
import math
import os

import numpy as np
import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel

from .common import OUT, cache_write, digest, load, model_lock, read, save_json
from .data import batch_supervision, keys_index, load_dataset, supervision
from .model import Gain, center, gain_input
from .bundle import construct
from .inputs import batch
from .targets import masks, weights
from graphllm_local_message_unique_v2.bridge import qualified


def loss_terms(model,result,targets,edge,stage):
    y=targets["y"]
    logits=result if stage=="self" else result["final"]
    sem=F.cross_entropy(logits.float(),y,ignore_index=-1,reduction="none")
    out=dict(sem=(sem*targets["sem_weight"]).sum())
    if stage=="self":
        return out
    _,_,bad,all_bad=masks(y,targets["relation"],edge)
    rel=F.cross_entropy(result["relation"].float().transpose(1,2),targets["relation"],
                        ignore_index=-1,reduction="none")
    out["rel"]=(rel*targets["rel_weight"]).sum()
    count=edge.sum(1).clamp_min(1)
    alternative=result["message"][:,None]-result["t"]/count[:,None,None]
    head=model.evidence
    removed=result["base"][:,None]+head.correct(alternative,result["base_hidden"][:,None])
    err=result["t"].square().mean(-1)+(center(result["final"])[:,None]-center(removed).detach()).square().mean(-1)
    base_err=center(result["delta"]).square().mean(-1)
    value=((err*bad).sum(1)+all_bad*base_err)/(bad.sum(1)+all_bad).clamp_min(1)
    out["bad"]=(value*targets["bad_weight"]).sum()
    return out


def module_for(model,stage):
    return model.receiver if stage=="self" else model.evidence


def optimizer(model,stage,cfg):
    from collections import defaultdict

    groups_=defaultdict(list)
    for name,p in module_for(model,stage).named_parameters():
        assert p.requires_grad
        head=name.startswith(("head.","relation.","wm.","wo.","wb."))
        decay=p.ndim>1 and not name.startswith("role")
        groups_[head,decay].append(p)
    t=cfg["training"]
    settings=[dict(params=ps,lr=t["head_lr"] if head else t["adapter_lr"],
                   weight_decay=t["weight_decay"] if decay else 0.) for (head,decay),ps in groups_.items()]
    return torch.optim.AdamW(settings)


def state_of(model,stage):
    return {k:v.detach().cpu().clone() for k,v in module_for(model,stage).state_dict().items()}


def fingerprint(model,stage):
    import hashlib

    sha=hashlib.sha256()
    mods=[model.qwen]
    if stage=="message":
        mods.append(model.receiver)
    for module in mods:
        for name,p in module.state_dict().items():
            sha.update(name.encode())
            sha.update(p.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return sha.hexdigest()


def run_batch(model,wrapped,items,ci,cfg,stage,device,split_cache=None):
    b=batch([r["item"] for r in items],ci,cfg,device)
    t=batch_supervision(items,b["edge_mask"].shape[1],device)
    anchor=None
    if split_cache is not None:
        anchor={k:v[[r["base_index"] for r in items]].to(device).float() for k,v in split_cache.items()}
    result=wrapped(b,stage=stage,base=anchor)
    if stage=="message":
        result["base_hidden"]=anchor["hidden"]
    terms=loss_terms(model,result,t,b["edge_mask"],stage)
    return result,t,terms


@torch.inference_mode()
def base_records(model,rows,ci,cfg,device):
    unique={}
    indices=[]
    for r in rows:
        key=r["item"][0]["key"]
        if key not in unique:
            unique[key]=r
        indices.append(key)
    keys=list(unique)
    bank={}
    for i in range(0,len(keys),64):
        group=[unique[k]["item"] for k in keys[i:i+64]]
        out=model.read_base(batch(group,ci,cfg,device))
        for k,v in out.items():
            bank.setdefault(k,[]).append(v.to("cpu",dtype=model.qwen.dtype) if k=="prefix" else v.cpu())
    index={k:i for i,k in enumerate(keys)}
    for r,k in zip(rows,indices):
        r["base_index"]=index[k]
    return {k:torch.cat(v) for k,v in bank.items()}


def base_cache(model,rows,ci,cfg,device,split,lock,rank,world):
    # Stage A snapshot is immutable. Each rank can reopen the same cache.
    path=OUT/f"self_cache_{split}.pt"
    keys=list(dict.fromkeys(r["item"][0]["key"] for r in rows))
    if rank==0 and not path.exists():
        obj=base_records(model,rows,ci,cfg,device)
        cache_write(path,dict(lock=lock,keys=keys,records=obj))
    if world>1:
        dist.barrier()
    disk=torch.load(path,map_location="cpu",weights_only=True,mmap=True)
    assert disk["lock"]==lock and disk["keys"]==keys
    index={k:i for i,k in enumerate(keys)}
    for r in rows:
        r["base_index"]=index[r["item"][0]["key"]]
    return disk["records"]


@torch.no_grad()
def monitor(model,rows,ci,cfg,stage,device,rank,world,base):
    model.eval()
    acc=torch.zeros(3,device=device,dtype=torch.float64)
    confusion=torch.zeros((2,3,3),device=device,dtype=torch.long)
    motion=torch.zeros(5,device=device,dtype=torch.float64)
    ix=list(range(rank,len(rows),world))
    for i in range(0,len(ix),32):
        result,t,terms=run_batch(model,model,[rows[j] for j in ix[i:i+32]],ci,cfg,stage,device,base)
        acc+=torch.stack([terms.get(k,acc.new_zeros(())) for k in ("sem","rel","bad")]).double()
        sem=result if stage=="self" else result["final"]
        yy=t["y"]
        valid=(t["sem_weight"]>0)
        confusion[0]+=torch.bincount(3*yy[valid]+sem.argmax(1)[valid],minlength=9).reshape(3,3)
        if stage=="message":
            valid=t["rel_weight"]>0
            y=t["relation"][valid]
            confusion[1]+=torch.bincount(3*y+result["relation"].argmax(-1)[valid],minlength=9).reshape(3,3)
            base_pick=result["base"].argmax(1)
            final_pick=result["final"].argmax(1)
            labelled=t["sem_weight"]>0
            motion+=torch.stack(((base_pick!=final_pick).sum(),
                (labelled&(base_pick!=yy)&(final_pick==yy)).sum(),
                (labelled&(base_pick==yy)&(final_pick!=yy)).sum(),
                result["message"].norm(dim=1).sum(),result["delta"].norm(dim=1).sum())).double()
    if world>1:
        dist.all_reduce(acc)
        dist.all_reduce(confusion)
        dist.all_reduce(motion)
    assert torch.isfinite(acc).all()
    return dict(total=float(acc.sum()),sem=float(acc[0]),relation=float(acc[1]),bad=float(acc[2]),
                semantic_confusion=confusion[0].tolist(),relation_confusion=confusion[1].tolist(),
                message=dict(changed=int(motion[0]),rescued=int(motion[1]),degraded=int(motion[2]),
                             mean_norm=float(motion[3])/len(rows),mean_delta=float(motion[4])/len(rows)))


def output_probe(model,row,ci,cfg,stage,device,base):
    b=batch([row["item"]],ci,cfg,device)
    cached={k:v[row["base_index"]:row["base_index"]+1].to(device).float() for k,v in base.items()} if base else None
    with torch.no_grad():
        model.eval()
        out=model(b,stage=stage,base=cached)
        return (out if stage=="self" else torch.cat((out["final"],out["message"]),1)).cpu()


def train_graph(cfg,stage):
    world=int(os.environ.get("WORLD_SIZE","1"))
    rank=int(os.environ.get("RANK","0"))
    gpu=int(os.environ.get("LOCAL_RANK","0"))
    torch.cuda.set_device(gpu)
    device=torch.device("cuda",gpu)
    if world>1:
        dist.init_process_group("nccl")
    try:
        train_graph_worker(cfg,stage,rank,world,device)
    finally:
        if world>1:
            dist.destroy_process_group()


def train_graph_worker(cfg,stage,rank,world,device):
    lock=model_lock(stage)
    assert not (OUT/f"{stage}_qualification.json").exists()
    t=cfg["training"]
    full,micro=t["batch"],t["micro_batch"]
    assert full%(micro*world)==0
    train,mo=(supervision(*load_dataset(s)[:2],stage,cfg)[0] for s in ("train","monitor"))
    model=construct(cfg,device,stage)
    ci=keys_index(list(read(OUT/"data_lock.json")["vocabulary"]["text_to_canonical"]))
    before=fingerprint(model,stage)
    base={}
    if stage=="message":
        base_lock=dict(self=lock["self"],feature=read(OUT/"BUILD_COMPLETE.json")["feature_lock"])
        for name,rows in (("train",train),("monitor",mo)):
            base[name]=base_cache(model,rows,ci,cfg,device,name,base_lock,rank,world)
    opt=optimizer(model,stage,cfg)
    max_steps=math.ceil(len(train)/full)*t["epochs"]
    warm=math.ceil(max_steps*t["warmup"])
    def factor(step):
        if step<warm:
            return (step+1)/warm
        return .5*(1+math.cos(math.pi*(step-warm)/max(1,max_steps-warm)))
    schedule=torch.optim.lr_scheduler.LambdaLR(opt,factor)
    wrapped=DistributedDataParallel(model,device_ids=[device.index],broadcast_buffers=False) if world>1 else model
    progress=OUT/f"{stage}_resume.pt"
    best_path=OUT/f"{stage}_best.pt"
    def checkpoint(path,point,metric):
        cache_write(path,dict(lock=lock,point=point,monitor=metric,state=state_of(model,stage),
                              probe=output_probe(model,mo[0],ci,cfg,stage,device,base.get("monitor"))))
    if progress.exists():
        disk=load(progress)
        assert disk["lock"]==lock and disk["world"]==world
        module_for(model,stage).load_state_dict(disk["state"],strict=True)
        opt.load_state_dict(disk["optimizer"])
        schedule.load_state_dict(disk["scheduler"])
        history=disk["history"]
        best,best_point=disk["best"],disk["best_point"]
    else:
        initial=monitor(model,mo,ci,cfg,stage,device,rank,world,base.get("monitor"))
        history=[dict(point=0,monitor=initial,train=None,lr=schedule.get_last_lr())]
        best,best_point=initial["total"],0
        if rank==0:
            checkpoint(OUT/f"{stage}_initial.pt",0,initial)
            checkpoint(best_path,0,initial)
    for epoch in range(history[-1]["point"]+1,t["epochs"]+1):
        if epoch-1-best_point>=t["patience"]:
            break
        model.train()
        order=np.random.default_rng(cfg["seed"]+epoch).permutation(len(train)).tolist()
        acc=torch.zeros(3,device=device,dtype=torch.float64)
        for i in range(0,len(order),full):
            selected=order[i:i+full]
            n=len(selected)
            padded=selected+[selected[-1]]*(full-n)
            opt.zero_grad(set_to_none=True)
            for start in range(rank*micro,full,micro*world):
                ids=padded[start:start+micro]
                real=max(0,min(micro,n-start))
                items=[train[x] for x in ids]
                if real<micro:
                    items=items[:real]+[dict(r,sem_weight=0.,rel_weight=[0.]*len(r["relation"]),bad_weight=0.)
                                       for r in items[real:]]
                sync=start+micro*world>=full
                with wrapped.no_sync() if world>1 and not sync else nullcontext():
                    _,_,terms=run_batch(model,wrapped,items,ci,cfg,stage,device,base.get("train"))
                    val=sum(terms.values())
                    if not torch.isfinite(val):
                        raise FloatingPointError(stage)
                    (val*world*len(train)/n).backward()
                acc+=torch.stack([terms.get(k,acc.new_zeros(())).detach() for k in ("sem","rel","bad")]).double()
            torch.nn.utils.clip_grad_norm_(module_for(model,stage).parameters(),t["clip"],error_if_nonfinite=True)
            opt.step()
            schedule.step()
        if world>1:
            dist.all_reduce(acc)
        metrics=monitor(model,mo,ci,cfg,stage,device,rank,world,base.get("monitor"))
        history.append(dict(point=epoch,train=dict(zip(("sem","relation","bad"),acc.tolist())),
                            monitor=metrics,lr=schedule.get_last_lr()))
        if metrics["total"]<best:
            best,best_point=metrics["total"],epoch
            if rank==0:
                checkpoint(best_path,epoch,metrics)
        if rank==0:
            cache_write(progress,dict(lock=lock,world=world,state=state_of(model,stage),optimizer=opt.state_dict(),
                        scheduler=schedule.state_dict(),history=history,best=best,best_point=best_point))
            save_json(OUT/f"{stage}_history.json",history)
            print(stage,epoch,history[-1],flush=True)
        if world>1:
            dist.barrier()
    assert before==fingerprint(model,stage)
    if rank==0:
        last=history[-1]
        checkpoint(OUT/f"{stage}_final.pt",last["point"],last["monitor"])
        disk=load(best_path)
        assert disk["point"]==best_point
        module_for(model,stage).load_state_dict(disk["state"],strict=True)
        module_for(model,stage).requires_grad_(False).eval()
        tolerance=.01 if model.qwen.dtype==torch.bfloat16 else 1e-5
        torch.testing.assert_close(output_probe(model,mo[0],ci,cfg,stage,device,base.get("monitor")),
                                   disk["probe"],atol=tolerance,rtol=tolerance)
        save_json(OUT/f"{stage}_qualification.json",dict(version=t["qualification"],completed=True,
            learning_supported=best_point>0,stop="patience" if last["point"]<t["epochs"] else "budget",
            initial=history[0]["monitor"],best=disk["monitor"],final=last["monitor"],best_point=best_point,
            points=last["point"],frozen_hash=before,probe_tolerance=tolerance,best_hash=digest(best_path),
            final_hash=digest(OUT/f"{stage}_final.pt"),lock=lock))


@torch.inference_mode()
def paired_gain_rows(cfg,model,risk,rows,ci,device):
    out=[]
    for i in range(0,len(rows),32):
        b=batch(rows[i:i+32],ci,cfg,device)
        result=model(b)
        native=b["native16"].cpu()
        rl=risk(native)
        msg=result["message"].cpu()
        full=gain_input(native,rl,result["final"].cpu(),msg)
        off=gain_input(native,rl,result["base"].cpu(),torch.zeros_like(msg))
        assert torch.isfinite(full).all() and torch.isfinite(off).all()
        out.append(torch.stack((full,off),1).cpu())
    return torch.cat(out)


def train_gain(cfg):
    lock=model_lock("gain")
    assert not (OUT/"gain_qualification.json").exists()
    device="cuda"
    model=construct(cfg,device,"frozen")
    risk,_=qualified()
    risk.requires_grad_(False)
    ci=keys_index(read(OUT/"data_lock.json")["vocabulary"]["text_to_canonical"])
    def get(split):
        path=OUT/f"gain_cache_{split}.pt"
        rows,labels,_=load_dataset(split)
        records=[r for r in rows if r[0]["view"]==0 and labels[r[0]["key"],0]["gain_target"]>=0]
        keys=[r[0]["key"] for r in records]
        if path.exists():
            data=load(path)
            assert data["lock"]==lock and data["keys"]==keys
            x=data["features"]
        else:
            x=paired_gain_rows(cfg,model,risk,records,ci,device)
            cache_write(path,dict(lock=lock,keys=keys,features=x))
        y=[labels[k,0]["gain_target"] for k in keys]
        w=weights(list(zip(y,keys)),(0,1))
        return x,torch.tensor(y,dtype=torch.float32),torch.tensor(w,dtype=torch.float64)
    a,y,w=get("train")
    b,v,u=get("monitor")
    assert digest(OUT/"self_best.pt")==lock["self"] and digest(OUT/"message_best.pt")==lock["message"]
    del model
    torch.manual_seed(cfg["seed"])
    head=Gain(cfg)
    a2=a.flatten(0,1)
    head.mean.copy_(a2.mean(0))
    head.std.copy_(a2.std(0,unbiased=False).clamp_min(cfg["normalizer"]["floor"]))
    settings=cfg["gain"]
    opt=torch.optim.AdamW([dict(params=[p for p in head.parameters() if p.ndim>1]),
                          dict(params=[p for p in head.parameters() if p.ndim==1],weight_decay=0.)],
                         lr=settings["lr"],weight_decay=settings["weight_decay"])
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,settings["steps"])
    def loss(x,y,w):
        logits=head(x.flatten(0,1)).reshape(-1,2)
        return (F.binary_cross_entropy_with_logits(logits,y[:,None].expand_as(logits),reduction="none").mean(1)*w).sum()
    def checkpoint(name,step,value):
        cache_write(OUT/f"gain_{name}.pt",dict(lock=lock,point=step,monitor=value,state=head.state_dict(),
                    probe=head(b[:8].flatten(0,1)).detach()))
    path=OUT/"gain_resume.pt"
    if path.exists():
        current=load(path)
        assert current["lock"]==lock
        head.load_state_dict(current["state"])
        opt.load_state_dict(current["optimizer"])
        schedule.load_state_dict(current["scheduler"])
        history,best,best_point=(current[k] for k in ("history","best","best_point"))
    else:
        best=float(loss(b,v,u))
        best_point=0
        history=[dict(point=0,train=float(loss(a,y,w)),monitor=best,lr=settings["lr"])]
        checkpoint("initial",0,best)
        checkpoint("best",0,best)
    for step in range(history[-1]["point"]+1,settings["steps"]+1):
        opt.zero_grad(set_to_none=True)
        value=loss(a,y,w)
        assert torch.isfinite(value)
        value.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True)
        opt.step()
        schedule.step()
        with torch.no_grad():
            m=float(loss(b,v,u))
            tr=float(loss(a,y,w))
        assert math.isfinite(m)
        history.append(dict(point=step,train=tr,monitor=m,lr=schedule.get_last_lr()[0]))
        if m<best:
            best,best_point=m,step
            checkpoint("best",step,m)
        cache_write(path,dict(lock=lock,history=history,best=best,best_point=best_point,state=head.state_dict(),
                    optimizer=opt.state_dict(),scheduler=schedule.state_dict()))
    checkpoint("final",history[-1]["point"],history[-1]["monitor"])
    disk=load(OUT/"gain_best.pt")
    head.load_state_dict(disk["state"])
    head.requires_grad_(False).eval()
    torch.testing.assert_close(head(b[:8].flatten(0,1)),disk["probe"],rtol=0,atol=0)
    save_json(OUT/"gain_history.json",history)
    save_json(OUT/"gain_qualification.json",dict(version=cfg["training"]["qualification"],completed=True,
        learning_supported=best_point>0,stop="budget",initial=history[0]["monitor"],best=best,
        final=history[-1]["monitor"],best_point=best_point,points=history[-1]["point"],
        best_hash=digest(OUT/"gain_best.pt"),final_hash=digest(OUT/"gain_final.pt"),lock=lock))
    print("gain",len(a),len(b),history[0],history[-1],best_point,flush=True)
