"""Protect original matches before choosing a fixed GraphLLM class update."""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from objects365_cache import block_lock, block_path, split_rows
from objects365_claim_cache import OUT as CLAIM, PLAN as CLAIM_PLAN
from objects365_data import BLOCK, FOLDER, INDEX, SHARDS, digest, read, save_json
from objects365_donor_view_entropy import OUT as GRAPH
from objects365_donor_view_entropy_eval import entropy, measure, prefix_curve
from objects365_edge_cache import split_document
from objects365_fit import qualified as base_qualified, replay_metrics
from objects365_receiver_native_evidence import OUT as NATIVE
from objects365_rare import RARE_INDEX

OUT = FOLDER/"receiver_keep_gate"
PLAN = Path(__file__).with_name("OBJECTS365_RECEIVER_KEEP_GATE.md")
SEED = 20261112
EPOCHS = 120
UNKNOWN_BUDGET = {"strict":0,"limited":11}
QUANTILES = (.01,.05,.10,.25,.50)
ARMS = ("mean","mixed_entropy")


def anchor():
    return dict(index=digest(INDEX),rare=digest(RARE_INDEX),plan=digest(PLAN),code=digest(Path(__file__)),
                native=digest(NATIVE/"qualification.json"),claim=digest(CLAIM_PLAN),
                graph=digest(GRAPH/"result.json"))


def read_rows(doc,split):
    source,original,index = split_document(doc,split)
    for shard in range(SHARDS):
        rows = split_rows(source,original,shard)
        for offset in range(0,len(rows),BLOCK):
            group = rows[offset:offset+BLOCK]
            part = offset//BLOCK
            name = f"{split}_{shard}_{part:03d}.pt"
            path = block_path(split,shard,part)
            cached = torch.load(path,map_location="cpu",weights_only=True)
            assert cached["lock"]==block_lock(index,group)
            claims = torch.load(CLAIM/name,map_location="cpu",weights_only=True)
            assert claims["key"]["method"]==digest(CLAIM_PLAN)
            edge_lock = claims["key"]["context"]["edge"]
            assert all(edge_lock[k]==v for k,v in cached["lock"].items())
            yield group,cached["records"],claims["features"],name,path


def matched(row,claim):
    if claim is None:
        return torch.empty(0,dtype=torch.long),None
    lookup = {tuple(action):i for i,action in enumerate(claim["actions"].tolist())}
    pairs = [(i,lookup[key]) for i,action in enumerate(row["actions"].tolist())
             if (key:=tuple(action)) in lookup]
    if not pairs:
        return torch.empty(0,dtype=torch.long),None
    idx,other = torch.tensor(pairs).T
    assert torch.equal(row["actions"][idx],claim["actions"][other])
    assert torch.equal(row["actions"][idx,1],claim["labels"][other,1])
    return idx,claim["x"][other].float()


def features(x):
    native = x[:,0,:,:4].clamp(.001,.999).logit().flatten(1)
    visual = x[:,:,:,4:]
    mean = visual.mean(1).flatten(1)
    spread = (visual[:,0]-visual[:,1]).abs().flatten(1)
    return torch.cat((native,mean,spread),1)


def harmful(outcome):
    return bool(outcome["wrong"] or any(outcome["lost"]))


@torch.no_grad()
def dataset(doc,splits,proposer,receipt):
    x,y,images,proposal = [],[],[],[]
    counts = Counter()
    for split in splits:
        for group,records,claims,_,_ in read_rows(doc,split):
            counts["images"] += len(group)
            for row in records:
                if "actions" not in row:
                    continue
                idx,raw = matched(row,claims.get(row["image"]))
                counts.update(eligible=len(row["actions"]), matched=len(idx))
                if not len(idx):
                    continue
                rows = [row["outcomes"][i] for i in idx]
                seen = torch.tensor([o["target"]>=0 for o in rows])
                if not seen.any():
                    continue
                h = row["x"][idx[seen],0].float().to(next(proposer.parameters()).device)
                logits = proposer(h)
                margin = raw[seen,0,1,0].clamp(.001,.999).logit()-raw[seen,0,0,0].clamp(.001,.999).logit()
                score = (logits[:,1]-logits[:,0]).cpu()+receipt["best_weight"]*((margin-receipt["center"])/receipt["scale"])
                proposal.append(score)
                x.append(features(raw[seen]))
                y.extend([harmful(o) for o in rows if o["target"]>=0])
                images.extend([row["image"]]*int(seen.sum()))
    x,y,score = torch.cat(x),torch.tensor(y),torch.cat(proposal)
    by_image = torch.unique(torch.tensor(images),return_inverse=True)[1]
    size = int(by_image.max())+1
    w = torch.zeros(len(y))
    for label in (False,True):
        use = y==label
        n = torch.bincount(by_image[use],minlength=size)
        assert use.any()
        w[use] = .5/((n>0).sum()*n[by_image[use]])
        hard = use & (score>=torch.quantile(score[use],.75))
        h = torch.bincount(by_image[hard],minlength=size)
        w[use] *= .75
        w[hard] += .125/((h>0).sum()*h[by_image[hard]])
    assert abs(float(w.sum())-1)<1e-5
    counts.update(labeled=len(y),harm=int(y.sum()),labeled_images=size)
    return (x,y.float(),by_image,w,size),dict(counts),set(images)


class KeepRisk(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("center",torch.zeros(16))
        self.register_buffer("scale",torch.ones(16))
        self.net = nn.Sequential(nn.Linear(16,16),nn.GELU(),nn.Linear(16,1))

    def forward(self,x):
        return self.net(((x-self.center)/self.scale).clamp(-6,6)).flatten()


def losses(model,data):
    x,y,image,w,n = data
    terms = F.binary_cross_entropy_with_logits(model(x),y,reduction="none")
    return terms.new_zeros(n).scatter_add_(0,image,terms*w)*n


@torch.no_grad()
def measured(model,data):
    v = losses(model,data)
    assert v.isfinite().all()
    return float(v.mean()),v.cpu().numpy()


def save_model(model,step,name,lock):
    path = OUT/f"{name}.pt"
    tmp = path.with_suffix(".part")
    torch.save(dict(model=model.state_dict(),step=step,anchor=lock),tmp)
    tmp.replace(path)
    return digest(path)


def fit(doc):
    lock = anchor()
    receipt = read(NATIVE/"qualification.json")
    assert receipt["passed"]
    proposer,_ = base_qualified()
    training,tc,train_ids = dataset(doc,("train","rare_train"),proposer,receipt)
    monitor,mc,monitor_ids = dataset(doc,("monitor",),proposer,receipt)
    assert not (train_ids&monitor_ids)
    del proposer
    torch.manual_seed(SEED)
    model = KeepRisk()
    model.center.copy_(training[0].mean(0))
    model.scale.copy_(training[0].std(0).clamp_min(.01))
    nn.init.zeros_(model.net[-1].weight)
    nn.init.zeros_(model.net[-1].bias)
    initial,values = measured(model,monitor)
    draws = np.random.default_rng(SEED).integers(0,len(values),(2000,len(values)))
    a,b = np.quantile(values[draws].mean(1),(.025,.975))
    tol = float(max(.005,(b-a)/2))
    optimizer = torch.optim.AdamW(model.parameters(),lr=.015,weight_decay=.05)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=EPOCHS)
    best_value,best_step = initial,0
    best_hash = save_model(model,0,"best",lock)
    history = []
    for step in range(1,EPOCHS+1):
        optimizer.zero_grad(set_to_none=True)
        loss = losses(model,training).mean()
        assert loss.isfinite()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        schedule.step()
        tr,_ = measured(model,training)
        valid,_ = measured(model,monitor)
        history.append(dict(step=step,train=tr,monitor=valid,lr=schedule.get_last_lr()[0]))
        if valid<best_value:
            best_value,best_step = valid,step
            best_hash = save_model(model,step,"best",lock)
    final_hash = save_model(model,EPOCHS,"final",lock)
    last = np.array([x["monitor"] for x in history[-5:]])
    t = [x["train"] for x in history[-5:]]
    checks = dict(schedule=history[-1]["lr"]==0,plateau=abs(t[-1]-t[0])/t[0]<=.025,
                  slope=abs(np.polyfit(np.arange(5),last,1)[0])<=tol/4,
                  spread=float(np.ptp(last))<=tol,gap=last[-1]-best_value<=tol)
    state = torch.load(OUT/"best.pt",map_location="cpu",weights_only=True)
    model.load_state_dict(state["model"])
    with torch.no_grad():
        risk = model(monitor[0]).sigmoid()
        caps = [float(torch.quantile(risk[monitor[1]>0],q)) for q in QUANTILES]+[1.]
        y = monitor[1].numpy().astype(int)
        from ownership_graph_audit import auc
        auroc = float(auc(risk.numpy(),y))
    result = dict(anchor=lock,passed=all(checks.values()) and best_step>0,
                  checks={k:bool(v) for k,v in checks.items()},initial=initial,best=best_value,
                  best_step=best_step,final=history[-1],best_hash=best_hash,final_hash=final_hash,
                  monitor_auc=auroc,caps=caps,train=tc,monitor=mc,tolerance=tol)
    save_json(OUT/"history.json",history)
    save_json(OUT/"qualification.json",result)
    print("Qualification",result,flush=True)


def qualified():
    receipt = read(OUT/"qualification.json")
    assert receipt["passed"] and receipt["anchor"]==anchor() and digest(OUT/"best.pt")==receipt["best_hash"]
    state = torch.load(OUT/"best.pt",map_location="cpu",weights_only=True)
    assert state["anchor"]==receipt["anchor"] and state["step"]==receipt["best_step"]
    model = KeepRisk().eval()
    model.load_state_dict(state["model"])
    return model,receipt


@torch.inference_mode()
def evidence(doc,split,model):
    settings = read(NATIVE/"qualification.json")
    data,original,counts = {},[],Counter()
    for frames,records,claims,name,path in read_rows(doc,split):
        views = torch.load(GRAPH/name,map_location="cpu",weights_only=True)
        native = torch.load(NATIVE/name,map_location="cpu",weights_only=True)
        images = [r["id"] for r in frames]
        assert views["lock"]["source"]==digest(path) and views["lock"]["images"]==images
        assert native["lock"]["anchor"]==settings["anchor"] and native["lock"]["view"]==digest(GRAPH/name)
        original.extend(records)
        counts["images"] += len(frames)
        for row in records:
            if "actions" not in row:
                continue
            iid = row["image"]
            raw = views["records"][iid]
            margin = native["records"][iid].double()
            assert torch.equal(row["actions"],raw["actions"])
            p = raw["views"].double().clamp(1e-7,1-1e-7)
            p = (p.logit()+settings["best_weight"]*((margin-settings["center"])/settings["scale"])[:,None]).sigmoid()
            mean = p.mean(1)
            js = (entropy(mean)-entropy(p).mean(1)).clamp_min(0)
            multi = set(raw["selected"].tolist())
            claim = claims.get(iid)
            indices,x = matched(row,claim)
            if len(indices):
                assert torch.allclose(x[:,0,0,0],raw["score"][indices],atol=.00015,rtol=0)
            risk = torch.full((len(mean),),float("inf"))
            if len(indices):
                risk[indices] = model(features(x)).sigmoid()
            out = row["outcomes"]
            counts.update(candidate_images=1,actions=len(out),matched=len(indices),
                          possible=int(any(o["correct"] for o in out)),
                          gain_possible=int(any(o["correct"] and (o["new"][0] or o["new"][3]) for o in out)))
            q,c,s = row["actions"].T
            data[iid] = dict(outcomes=out,query=q,proposal=c,slot=s,mean=mean,risk=risk,js=js,multi=multi)
    return data,original,dict(counts)


def winners(data,cap,js_cap=None):
    rows = []
    for iid,item in data.items():
        score = item["mean"].masked_fill(item["risk"]>cap,-torch.inf)
        if js_cap is not None:
            eligible = torch.ones(len(score),dtype=torch.bool)
            positions = sorted(item["multi"])
            eligible[positions] = item["js"][positions]<=js_cap
            score.masked_fill_(~eligible,-torch.inf)
        j = int(score.argmax())
        if not torch.isfinite(score[j]):
            continue
        outcome = item["outcomes"][j]
        rows.append(dict(outcome,image=iid,query=int(item["query"][j]),proposed=int(item["proposal"][j]),
                         slot=int(item["slot"][j]),score=float(score[j]),risk=float(item["risk"][j]),
                         js=float(item["js"][j]) if j in item["multi"] else None,harmful=harmful(outcome)))
    return rows


def objective(row):
    return (*row["new"],row["correct"],-row["unknown"],-row["adopted"])


def select(doc,model,receipt):
    data,_,counts = evidence(doc,"policy",model)
    trials = []
    for cap in receipt["caps"]:
        rows = winners(data,cap)
        supported = [r["js"] for r in rows if r["correct"] and r["score"]>.5 and r["js"] is not None]
        jscap = float(np.quantile(supported,.90)) if supported else None
        for arm in ARMS:
            j = jscap if arm=="mixed_entropy" else None
            picked = winners(data,cap,j)
            curve = prefix_curve(picked,None,counts)
            for budget,unknown in UNKNOWN_BUDGET.items():
                safe = [r for r in curve if not (r["harmful"] or r["loss_images"]) and r["unknown"]<=unknown]
                point = max(safe,key=objective)
                trials.append(dict(arm=arm,budget=budget,risk_cap=cap,js_cap=j,point=point))
    points = {arm:{b:max((r for r in trials if r["arm"]==arm and r["budget"]==b),
                         key=lambda r:objective(r["point"])) for b in UNKNOWN_BUDGET} for arm in ARMS}
    result = dict(qualification=digest(OUT/"qualification.json"),counts=counts,points=points,trials=trials)
    save_json(OUT/"policy.json",result)
    print("Policy",points,flush=True)


def evaluate(doc,model,receipt):
    policy = read(OUT/"policy.json")
    assert policy["qualification"]==digest(OUT/"qualification.json")
    data,native,counts = evidence(doc,"eval",model)
    selected,outcomes,frontier = {},{},{}
    for arm,budgets in policy["points"].items():
        for budget,setting in budgets.items():
            key = f"{arm}_{budget}"
            rows = winners(data,setting["risk_cap"],setting["js_cap"])
            cutoff = setting["point"]["cutoff"]
            selected[key] = [] if cutoff is None else [r for r in rows if r["score"]>=cutoff]
            outcomes[key] = measure(selected[key],counts["possible"],counts["candidate_images"])
            curve = prefix_curve(rows,None,counts)
            frontier[key] = max((r for r in curve if not r["harmful"] and r["unknown"]<=UNKNOWN_BUDGET[budget]),
                                key=objective)
    ap,classes = replay_metrics(doc,native,selected)
    result = dict(anchor=receipt["anchor"],policy=digest(OUT/"policy.json"),counts=counts,
                  outcomes=outcomes,frontier=frontier,ap=ap,classes=classes,selected=selected)
    save_json(OUT/"result.json",result)
    print("Result",counts,outcomes,"AP",ap,flush=True)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage",choices=("fit","policy","eval"))
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(6)
    doc = read(INDEX)
    if args.stage=="fit":
        fit(doc)
    else:
        model,receipt = qualified()
        (select if args.stage=="policy" else evaluate)(doc,model,receipt)


