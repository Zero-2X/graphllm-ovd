"""One semantic run and one 18-feature gain fit."""
from collections import defaultdict
from contextlib import nullcontext
import math

import numpy as np
import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel

from .bridge import FOLDER, OUT, cache_write, digest, load, read, read_model, save_json
from .core import Gain, fuse, gain_features, pack_actions, semantic_loss


def base_lock():
    return digest(OUT/"resolved_config.json")


def records(split):
    out = []
    source = FOLDER/"graphllm_local_message_v1"
    receipt = read(OUT/"build_receipt.json")
    lock = receipt["visual_source"]
    assert digest(source/"resolved_config.json")==lock
    hashes = receipt["cache_hashes"]
    for name in sorted(n for n in hashes if n.startswith(f"visual_{split}_")):
        path = source/name
        assert digest(path)==hashes[path.name]
        block = load(path)
        assert block["lock"]==lock
        out.extend(block["actions"])
    assert out,split
    return out


def groups(rows, field, states):
    table = {s:defaultdict(list) for s in range(states)}
    for i,row in enumerate(rows):
        table[row[field]][row["key"][0]].append(i)
    assert all(table.values()), {s:len(g) for s,g in table.items()}
    weights = torch.zeros(len(rows),dtype=torch.float64)
    for by_image in table.values():
        for ids in by_image.values():
            weights[ids] = 1/(states*len(by_image)*len(ids))
    assert abs(float(weights.sum())-1)<1e-10
    return table,weights


def weighted_epoch(n,seed):
    return np.random.default_rng(seed).permutation(n).tolist()


def tolerance(values, weights, ids, cfg):
    _,ix = np.unique(ids,return_inverse=True)
    w = np.asarray(weights,dtype=np.float64)
    sums = np.bincount(ix,w*np.asarray(values))
    masses = np.bincount(ix,w)
    rng = np.random.default_rng(cfg["seed"])
    take = rng.integers(len(masses),size=(cfg["convergence"]["bootstrap"],len(masses)))
    interval = np.quantile(sums[take].sum(1)/masses[take].sum(1),(.025,.975))
    return float(max(cfg["convergence"]["tolerance_floor"],(interval[1]-interval[0])/2))


def qualification(history,best,tol,schedule_finished,cfg):
    window = cfg["convergence"]["window"]
    tail = history[-window:]
    checks = dict(schedule=schedule_finished,finite=all(math.isfinite(r[k]) for r in history for k in ("train","monitor")),
                  window=len(tail)==window)
    if checks["window"]:
        train = [r["train"] for r in tail]
        valid = [r["monitor"] for r in tail]
        checks.update(plateau=abs(train[-1]-train[0])/max(1e-8,train[0])<=cfg["convergence"]["plateau_relative"],
                      slope=abs(np.polyfit(np.arange(window),valid,1)[0])<=tol/4,
                      spread=np.ptp(valid)<=tol,gap=valid[-1]-best<=tol)
    checks = {k:bool(v) for k,v in checks.items()}
    return dict(passed=all(checks.values()),checks=checks,tolerance=tol)


def make_sem_optimizer(model,cfg):
    settings = cfg["semantic"]
    buckets = defaultdict(list)
    for name,p in model.named_parameters():
        if p.requires_grad:
            head = name.startswith("head.")
            decay = p.ndim>1
            buckets[head,decay].append(p)
    return torch.optim.AdamW([dict(params=ps,lr=settings["head_lr" if head else "adapter_lr"],
                                    weight_decay=settings["weight_decay"] if decay else 0.)
                              for (head,decay),ps in buckets.items()])


def forward_loss(model,batch,cfg,training):
    device = next(model.parameters()).device
    data = pack_actions(batch,device)
    y = torch.tensor([r["semantic_target"] for r in batch],device=device)
    with torch.autocast("cuda",dtype=torch.bfloat16,enabled=torch.cuda.is_bf16_supported()):
        z,d = model(data)
    f = fuse(z,d,data["rho"],cfg["message"],training)["fused"]
    loss = semantic_loss(z,d,f,data["rho"],y)
    if not loss.isfinite().all():
        raise FloatingPointError([r["key"] for r in batch])
    return loss if training else (loss,f.argmax(-1)==y)


@torch.no_grad()
def validation(model,rows,cfg,rank,world):
    model.eval()
    device = next(model.parameters()).device
    values = torch.zeros((2,len(rows)),device=device)
    own = list(range(rank,len(rows),world))
    for start in range(0,len(own),32):
        ix = own[start:start+32]
        losses, correct = forward_loss(model,[rows[i] for i in ix],cfg,False)
        values[0,ix] = losses
        values[1,ix] = correct.to(values.dtype)
    if world>1:
        dist.all_reduce(values)
    return values.cpu().double()


def train_semantic(doc,cfg,rank=0,world=1):
    lock = base_lock()
    receipt = OUT/"semantic_qualification.json"
    if receipt.exists():
        raise RuntimeError("The one semantic run is already terminal")
    resolved = read(OUT/"resolved_config.json")
    settings = cfg["semantic"]
    micro = settings["micro_batch"]
    batch = settings["batch"]
    assert batch%(micro*world)==0
    train = [r for r in records("train") if r["semantic_target"]>=0]
    monitor = [r for r in records("monitor") if r["semantic_target"]>=0]
    monitor_labels = torch.tensor([r["semantic_target"] for r in monitor])
    _,train_weights = groups(train,"semantic_target",3)
    _,weights = groups(monitor,"semantic_target",3)
    model = read_model(doc,cfg,resolved)
    norm = load(OUT/"native5_train.pt")
    assert norm["lock"]==lock
    model.native_mean.copy_(norm["mean"])
    model.native_std.copy_(norm["std"])
    optimizer = make_sem_optimizer(model,cfg)
    steps = math.ceil(len(train)/batch)*settings["epochs"]
    warm = math.ceil(steps*settings["warmup"])
    def lr(k):
        if k<warm:
            return (k+1)/warm
        return .5*(1+math.cos(math.pi*min(1,(k-warm)/max(1,steps-warm))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lr)
    wrapped = DistributedDataParallel(model,device_ids=[torch.cuda.current_device()],broadcast_buffers=False) if world>1 else model
    progress = OUT/"semantic_resume.pt"
    if progress.exists():
        state = load(progress)
        assert state["lock"]==lock and state["world"]==world
        model.restore(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        history,best,best_epoch,tol = (state[k] for k in ("history","best","best_epoch","tolerance"))
    else:
        initial,_ = validation(model,monitor,cfg,rank,world)
        tol = tolerance(initial,weights,[r["key"][0] for r in monitor],cfg)
        history,best,best_epoch = [],float("inf"),0
    for epoch in range(len(history)+1,settings["epochs"]+1):
        if history and len(history)-best_epoch>=settings["patience"]:
            break
        model.train()
        draws = weighted_epoch(len(train),cfg["seed"]+epoch)
        total = torch.zeros((),device="cuda")
        for start in range(0,len(draws),batch):
            actual = draws[start:start+batch]
            # Every replica participates in every collective. Padded terms carry zero weight.
            padded = actual+[actual[-1]]*(batch-len(actual))
            optimizer.zero_grad(set_to_none=True)
            for pos in range(rank*micro,batch,micro*world):
                ix = padded[pos:pos+micro]
                count = max(0,min(micro,len(actual)-pos))
                sync = pos+micro*world>=batch
                context = wrapped.no_sync() if world>1 and not sync else nullcontext()
                with context:
                    terms = forward_loss(wrapped,[train[i] for i in ix],cfg,True)
                    weights_here = train_weights[ix[:count]].to(terms.device, terms.dtype)
                    contribution = (terms[:count]*weights_here).sum()
                    (contribution*world*len(draws)/len(actual)).backward()
                total += contribution.detach()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                           settings["grad_clip"],error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
        if world>1:
            dist.all_reduce(total)
        valid,correct = validation(model,monitor,cfg,rank,world)
        monitored = float((valid*weights).sum())
        recall = [float((correct[monitor_labels==i]*weights[monitor_labels==i]).sum()*3) for i in range(3)]
        history.append(dict(epoch=epoch,train=float(total),monitor=monitored,
                            lr=scheduler.get_last_lr()[0],recall=recall))
        if monitored<best:
            best,best_epoch = monitored,epoch
            if rank==0:
                cache_write(OUT/"semantic_best.pt",dict(lock=lock,model=model.artifact(),epoch=epoch))
        if rank==0:
            cache_write(progress,dict(lock=lock,world=world,model=model.artifact(),optimizer=optimizer.state_dict(),
                                     scheduler=scheduler.state_dict(),history=history,best=best,best_epoch=best_epoch,tolerance=tol))
            save_json(OUT/"semantic_history.json",history)
            print(history[-1],flush=True)
        if world>1:
            dist.barrier()
    if rank==0:
        cache_write(OUT/"semantic_final.pt",dict(lock=lock,model=model.artifact(),epoch=len(history)))
        q = qualification(history,best,tol,len(history)==settings["epochs"] and history[-1]["lr"]==0.,cfg)
        q.update(lock=lock,best=best,best_epoch=best_epoch,epochs=len(history),
                 best_hash=digest(OUT/"semantic_best.pt"),final_hash=digest(OUT/"semantic_final.pt"))
        save_json(receipt,q)
        print(q,flush=True)


def fitted_reader(doc,cfg):
    receipt = read(OUT/"semantic_qualification.json")
    path = OUT/"semantic_best.pt"
    assert digest(path)==receipt["best_hash"] and receipt["lock"]==base_lock()
    state = load(path)
    assert state["lock"]==base_lock() and state["epoch"]==receipt["best_epoch"]
    model = read_model(doc,cfg,read(OUT/"resolved_config.json"))
    model.restore(state["model"])
    return model.requires_grad_(False).eval()


@torch.inference_mode()
def semantic_records(model,rows,cfg):
    output = []
    for start in range(0,len(rows),32):
        group = rows[start:start+32]
        data = pack_actions(group,"cuda")
        with torch.autocast("cuda",dtype=torch.bfloat16,enabled=torch.cuda.is_bf16_supported()):
            z,d = model(data)
        if not z.isfinite().all() or not d.isfinite().all():
            raise FloatingPointError([r["key"] for r in group])
        f = fuse(z,d,data["rho"],cfg["message"],False)["fused"]
        x = gain_features(data["native16"],f).cpu()
        for i,r in enumerate(group):
            output.append(dict(key=r["key"],features=x[i].clone(),z_self=z[i].cpu(),z_donor=d[i].cpu(),
                               rho=r["rho"],utility=r["utility"],gain_target=r["gain_target"]))
    return output


def train_gain(doc,cfg):
    lock = base_lock()
    semantic_hash = digest(OUT/"semantic_best.pt")
    receipt = OUT/"gain_qualification.json"
    if receipt.exists():
        raise RuntimeError("The one gain fit is already terminal")
    model = fitted_reader(doc,cfg)
    for split in ("train","monitor","policy"):
        path = OUT/f"semantics_{split}.pt"
        if path.exists():
            cached=load(path)
            assert cached["lock"]==lock and cached["semantic"]==semantic_hash
            continue
        rows = records(split)
        if split!="policy":
            rows=[r for r in rows if r["gain_target"]>=0]
        values=semantic_records(model,rows,cfg)
        cache_write(path,dict(lock=lock,semantic=semantic_hash,rows=values))
        print("Frozen semantic records",split,len(values),flush=True)
    del model
    tr,mo = (load(OUT/f"semantics_{split}.pt")["rows"] for split in ("train","monitor"))
    _,tw = groups(tr,"gain_target",2)
    _,mw = groups(mo,"gain_target",2)
    def tensors(rows,w):
        return torch.stack([r["features"] for r in rows]),torch.tensor([r["gain_target"] for r in rows]).float(),w.float()
    a,y,w=tensors(tr,tw)
    b,v,u=tensors(mo,mw)
    gain=Gain()
    gain.mean.copy_(a.mean(0))
    gain.std.copy_(a.std(0,unbiased=False).clamp_min(.01))
    settings=cfg["gain"]
    opt=torch.optim.AdamW(gain.parameters(),lr=settings["lr"],weight_decay=settings["weight_decay"])
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,settings["steps"])
    def loss(x,y):
        return F.binary_cross_entropy_with_logits(gain(x),y,reduction="none")
    progress = OUT/"gain_resume.pt"
    if progress.exists():
        state = load(progress)
        assert state["lock"]==lock and state["semantic"]==semantic_hash
        gain.load_state_dict(state["model"],strict=True)
        opt.load_state_dict(state["optimizer"])
        schedule.load_state_dict(state["schedule"])
        history,best,best_step,tol=(state[k] for k in ("history","best","best_step","tolerance"))
    else:
        tol=tolerance(loss(b,v).detach(),mw,[r["key"][0] for r in mo],cfg)
        history,best,best_step=[],float("inf"),0
    for step in range(len(history)+1,settings["steps"]+1):
        opt.zero_grad(set_to_none=True)
        train=(loss(a,y)*w).sum()
        assert train.isfinite()
        train.backward()
        torch.nn.utils.clip_grad_norm_(gain.parameters(),settings["grad_clip"],error_if_nonfinite=True)
        opt.step()
        schedule.step()
        with torch.no_grad():
            valid=float((loss(b,v)*u).sum())
            history.append(dict(step=step,train=float((loss(a,y)*w).sum()),monitor=valid,lr=schedule.get_last_lr()[0]))
        state=dict(lock=lock,semantic=semantic_hash,model=gain.state_dict(),step=step)
        if valid<best:
            best,best_step=valid,step
            cache_write(OUT/"gain_best.pt",state)
        cache_write(progress,dict(state,history=history,best=best,best_step=best_step,tolerance=tol,
                                  optimizer=opt.state_dict(),schedule=schedule.state_dict()))
    cache_write(OUT/"gain_final.pt",dict(lock=lock,semantic=semantic_hash,model=gain.state_dict(),step=settings["steps"]))
    q=qualification(history,best,tol,len(history)==settings["steps"] and history[-1]["lr"]==0.,cfg)
    q.update(lock=lock,semantic=semantic_hash,best=best,best_step=best_step,best_hash=digest(OUT/"gain_best.pt"),
             final_hash=digest(OUT/"gain_final.pt"))
    save_json(OUT/"gain_history.json",history)
    save_json(receipt,q)
    print(q,flush=True)


def fitted_gain():
    q = read(OUT/"gain_qualification.json")
    path = OUT/"gain_best.pt"
    assert q["lock"]==base_lock() and digest(path)==q["best_hash"]
    state = load(path)
    assert state["lock"]==base_lock() and state["step"]==q["best_step"]
    assert state["semantic"]==q["semantic"]==digest(OUT/"semantic_best.pt")
    gain=Gain()
    gain.load_state_dict(state["model"],strict=True)
    return gain.requires_grad_(False).eval()
