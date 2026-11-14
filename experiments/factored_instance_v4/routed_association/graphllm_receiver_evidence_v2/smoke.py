"""Verify the actual detector, spatial path, frozen Qwen, labels and evaluator."""
import math

import torch
from torch.nn import functional as F

from .common import CODE, OUT, HISTORY, annotated_split, digest, load, read, safe_frame, save_json, vocabulary
from .inputs import batch
from .model import Gain, GraphReader, gain_input
from .native_bridge import NativeBridge, grid_xy, sample_local
from .predict import key_index, old_bundle, old_pick, one_slot, score_actions
from .targets import annotate, label_utility, masks, policy_prefix, relation_label, status, weights
from .train import fingerprint, loss_terms, optimizer


def spatial_checks(cfg):
    for h,w,vh,vw in ((10,14,8,11),(11,9,9,7),(8,8,8,8)):
        yy,xx=torch.meshgrid(torch.arange(h),torch.arange(w),indexing="ij")
        feature=torch.stack((xx.float(),yy.float(),2*xx.float()+yy))
        mask=(yy<vh)&(xx<vw)
        box=torch.tensor([[.20,.22,.76,.75],[.001,.001,.04,.05]])
        ratio=torch.tensor([vw/w,vh/h])
        result,ok=sample_local(feature,mask,ratio,box,3,cfg["local"]["min_weight"])
        xy=grid_xy(3)
        p=box[:,None,:2]+xy*(box[:,None,2:]-box[:,None,:2])
        raw=p*torch.tensor([vw,vh])-.5
        x,y=raw.clamp_min(0).unbind(-1)
        expect=torch.stack((x,y,2*x+y),-1)
        assert ok.all()
        torch.testing.assert_close(result,expect,atol=2e-5,rtol=0)
        _,invalid=sample_local(feature,mask,ratio,box*0-2,3,cfg["local"]["min_weight"])
        assert not invalid.any()


def label_checks():
    rec=lambda owner,y=1:dict(owner=owner,semantic=y,clear=True)
    ref=lambda owner,state:dict(owner=owner,source=state,clear=True)
    assert relation_label(rec(1),ref(2,"valid"))==0
    assert relation_label(rec(1,2),ref(2,"valid"))==1
    assert relation_label(rec(1),ref(2,"contradicted"))==2
    assert relation_label(rec(1),ref(1,"valid"))==2
    assert relation_label(rec(None),ref(2,"valid"))==-1
    e=torch.tensor([[True,True]]*5+[[False,False]])
    y=torch.tensor([1,1,1,1,1,-1])
    r=torch.tensor([[0,1],[1,2],[2,2],[0,-1],[2,-1],[-1,-1]])
    s,_,bad,all_bad=masks(y,r,e)
    assert s.tolist()==[True,True,False,False,False,False]
    assert all_bad.tolist()==[False,False,True,False,False,False] and not bad[0].any()
    empty=[set() for _ in range(6)]
    new=[{1},*empty[1:]]
    safe=label_utility(1,dict(target=1,correct=1,wrong=0),new,empty)
    bad=label_utility(-1,dict(target=-1,correct=0,wrong=0),new,new)
    unknown=label_utility(-1,dict(target=1,correct=1,wrong=0),new,empty)
    assert [safe["state"],bad["state"],unknown["state"]]==["safe_new","known_harm","unknown"]
    assert policy_prefix([(2,(1,),safe),(2,(2,),bad)],11)["gain_logit_cutoff"] is None
    assert policy_prefix([(2,(1,),safe),(1,(2,),bad)],11)["gain_logit_cutoff"]==2
    atoms=[(0,(1,1)),(0,(1,1)),(0,(1,2)),(0,(2,1)),(1,(2,2)),(2,(3,1))]
    w=weights(atoms,(0,1,2))
    assert math.isclose(sum(w[:4]),1/3) and w[0]==w[1]==w[2]/2==w[3]/4
    v=torch.arange(len(w),dtype=torch.float32)
    z=torch.tensor(w)
    torch.testing.assert_close((z*v).sum(),(z[:2]*v[:2]).sum()+(z[2:]*v[2:]).sum())


def execution_checks():
    from counterfactual import predictions
    from evaluate import coco_eval, metrics
    from .predict import select

    cats=[11,34,57]
    index=key_index(cats)
    base=dict(query=torch.tensor([0,1]),label=torch.tensor([0,2]),score=torch.tensor([.2,.7]),
              box=torch.tensor([[.0,.0,.3,.3],[.6,.6,.9,.9]]))
    raw=[(0,0,0,11,34),(0,0,0,11,57)]
    records=[dict(key=k,semantic_eligible=True,gain_logit=1.,image_top1=False,executed=False) for k in raw[::-1]]
    k,_=select(records,dict(gain_logit_cutoff=1.))
    assert k==raw[0] and sum(r["executed"] for r in records)==1
    changed=one_slot(base,k,index)
    assert changed["label"].tolist()==[1,2] and torch.equal(changed["score"],base["score"])
    frame=dict(id=0,file_name="synthetic",width=100,height=100,known=cats,annotations=[
        dict(id=1,image_id=0,bbox=[0,0,30,30],category_id=34,area=900,iscrowd=0),
        dict(id=2,image_id=0,bbox=[60,60,30,30],category_id=57,area=900,iscrowd=0)])
    doc=dict(base=["a","b","c"],names=["a","b","c"],categories=dict(zip(("a","b","c"),cats)))
    native=metrics(coco_eval([frame],predictions(frame,base,cats),cats))
    edited=metrics(coco_eval([frame],predictions(frame,changed,cats),cats))
    labels=annotate(frame,doc,dict(base=base),base["box"],{k:[]},store_matches=True)
    assert status(labels[k])=="safe_new" and labels[k]["new_ids"]==[[1],[1],[],[1],[1],[]]
    assert all(edited[v]>native[v] for v in ("AP","AP50","AP75"))
    duplicate=dict(base,label=torch.tensor([0,1]),query=torch.tensor([0,0]))
    try:
        one_slot(duplicate,k,index)
    except AssertionError:
        pass
    else:
        raise AssertionError("Duplicate query/class was accepted")


def check_gradient(p):
    assert p.grad is not None and p.grad.isfinite().all() and p.grad.count_nonzero()


def model_checks(cfg,doc,items,bridge):
    torch.manual_seed(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32=False
    model=GraphReader(doc,cfg,"cuda")
    assert not any(p.requires_grad for m in (model.qwen,bridge.vision.graph.native.model,bridge.vision.visual.clip,bridge.risk)
                   for p in m.parameters())
    ci=key_index(vocabulary(doc)["text_to_canonical"])
    b=batch(items,ci,cfg,"cuda")
    y=torch.arange(len(items),device="cuda")%3
    check=fingerprint(model,"self")
    aopt=optimizer(model,"self",cfg)
    before=model.receiver.global_[1].weight.detach().clone()
    logits=model(b,stage="self")
    F.cross_entropy(logits,y).backward()
    for p in (model.receiver.global_[1].weight,model.receiver.local[1].weight,model.receiver.head[-1].weight):
        check_gradient(p)
    aopt.step()
    assert not torch.equal(before,model.receiver.global_[1].weight)
    aopt.zero_grad(set_to_none=True)
    assert check==fingerprint(model,"self")
    model.add_evidence()
    e=model.evidence
    bopt=optimizer(model,"message",cfg)
    assert not ({id(p) for group in bopt.param_groups for p in group["params"]}&
                {id(p) for p in model.receiver.parameters()})
    before=fingerprint(model,"message")
    with torch.no_grad():
        base=model.read_base(b)
        read,i,j=model.read_relation(b,base,all_positions=True)
        tolerance=.01 if model.qwen.dtype==torch.bfloat16 else 1e-5
        prefix_only=model.read(base["prefix"][i])[:,-1]
        torch.testing.assert_close(read[:,base["prefix"].shape[1]-1],prefix_only,
                                   atol=tolerance,rtol=tolerance)
        prefix=torch.cat((base["prefix"][i],e.suffix(torch.zeros_like(b["donor"][:,:1])[i,0],
            torch.zeros_like(b["donor_local"][:,:1])[i,0],torch.zeros_like(b["edge"][:,:1])[i,0],
            base["label"][i],model.xy)),1)
        check=model.read(prefix)
        torch.testing.assert_close(check[:,base["prefix"].shape[1]-1],read[:,base["prefix"].shape[1]-1],
                                   atol=tolerance,rtol=tolerance)
        hidden=model.read_relation(b,base)
        same=e.apply(base,hidden[:,:1].expand_as(hidden),b["edge_mask"],model.xy)
        assert torch.count_nonzero(same["message"])==0
        identical=dict(base,phi=base["phi"][:,:1].expand_as(base["phi"]))
        assert torch.count_nonzero(e.apply(identical,hidden,b["edge_mask"],model.xy)["message"])==0
        assert torch.equal(e.correct(torch.zeros_like(same["message"]),base["hidden"]),torch.zeros_like(base["z"]))
        zero=e.apply(base,torch.zeros_like(hidden),b["edge_mask"],model.xy)
        torch.testing.assert_close(zero["nonnull"],torch.full_like(zero["nonnull"],.5))
    start=model(b,base=base)
    assert torch.equal(start["base"],start["final"]) and e.wm.weight.count_nonzero()
    F.cross_entropy(start["final"],y).backward()
    check_gradient(e.wo.weight)
    bopt.step()
    bopt.zero_grad(set_to_none=True)
    hidden=model.read_relation(b,base)
    hidden.retain_grad()
    out=e.apply(base,hidden,b["edge_mask"],model.xy)
    F.cross_entropy(out["final"],y).backward()
    for p in (e.query.weight,e.key.weight,e.value.weight,e.global_[1].weight,e.edge.weight):
        check_gradient(p)
    assert hidden.grad[:,0].count_nonzero()
    assert not any(p.grad is not None for p in model.receiver.parameters())
    bopt.zero_grad(set_to_none=True)
    out=model(b,base=base)
    F.cross_entropy(out["relation"][b["edge_mask"]],torch.zeros(int(b["edge_mask"].sum()),device="cuda",dtype=torch.long)).backward()
    check_gradient(e.relation[-1].weight)
    bopt.step()
    bopt.zero_grad(set_to_none=True)
    assert before==fingerprint(model,"message")
    with torch.no_grad():
        out=model(b,diagnose=True)
        torch.testing.assert_close(out["base"],base["z"],rtol=tolerance,atol=tolerance)
        off=model(b,off=True)
        assert torch.equal(off["base"],off["final"]) and not off["message"].count_nonzero()
        risk=bridge.risk(b["native16"].cpu())
        x=gain_input(b["native16"].cpu(),risk,off["final"].cpu(),off["message"].cpu())
        head=Gain(cfg)
        head.mean[19:].fill_(1)
        head.std[19:].fill_(2)
        assert x.shape[1]==147 and not x[:,19:].count_nonzero()
        assert ((x[:,19:]-head.mean[19:])/head.std[19:]==-.5).all()
        records=score_actions(items,b,out,risk,head,{},cfg,off=True)
        torch.testing.assert_close(torch.tensor([r["gain_logit"] for r in records]),head(x))
        target=dict(y=y,relation=torch.full_like(b["edge_mask"],-1,dtype=torch.long),
                    sem_weight=torch.zeros_like(y,dtype=torch.float32),
                    rel_weight=torch.zeros_like(b["edge_mask"],dtype=torch.float32),
                    bad_weight=torch.zeros_like(y,dtype=torch.float32))
        t=loss_terms(model,dict(out,base_hidden=base["hidden"]),target,b["edge_mask"],"message")
        assert all(float(v)==0 for v in t.values())
        # Removing one bad message in a mixed view keeps its original denominator.
        local=dict(hidden=base["hidden"][:1],z=base["z"][:1])
        t=torch.ones((1,2,128),device="cuda")
        t[:,0]*=.3
        t[:,1]*=.7
        avg=t.mean(1)
        delta=e.correct(avg,local["hidden"])
        actual=dict(base=local["z"],final=local["z"]+delta,message=avg,t=t,delta=delta,
                    base_hidden=local["hidden"],relation=torch.zeros(1,2,3,device="cuda"))
        labels=dict(y=torch.ones(1,dtype=torch.long,device="cuda"),
                    relation=torch.tensor([[0,2]],device="cuda"),sem_weight=torch.zeros(1,device="cuda"),
                    rel_weight=torch.zeros(1,2,device="cuda"),bad_weight=torch.ones(1,device="cuda"))
        got=loss_terms(model,actual,labels,torch.ones(1,2,dtype=torch.bool,device="cuda"),"message")["bad"]
        expect=t[:,1].square().mean()+(delta-e.correct(t[:,0]/2,local["hidden"])).square().mean()
        torch.testing.assert_close(got,expect)
        for g in (e.wo.weight,e.wm.weight):
            assert g.isfinite().all()
    return dict(frozen_hash=before,causal_tolerance=tolerance,qwen_dtype=str(model.qwen.dtype),
                self_parameters=sum(p.numel() for p in model.receiver.parameters()),
                message_parameters=sum(p.numel() for p in model.evidence.parameters()))


def replay_checks(doc,bridge,cfg):
    ci=key_index(bridge.categories)
    blind=read(HISTORY/"holdout_images.json")
    sealed=read(HISTORY/"predictions_manifest.json")
    old=old_bundle(doc,bridge.risk)
    chosen=[]
    for name,sha in sealed["files"].items():
        p=HISTORY/name
        assert digest(p)==sha
        for r in load(p)["rows"]:
            if r["selected"]["new_full"] is not None:
                chosen.append(r)
            if len(chosen)==3:
                break
        if len(chosen)==3:
            break
    assert len(chosen)==3
    by={f["id"]:f for f in blind["rows"]}
    items=[]
    for previous in chosen:
        iid=previous["image"]
        row,scan=bridge.scan(by[iid],training_pool=True)
        assert all(torch.equal(row["base"][k],previous["base"][k]) for k in row["base"])
        actual=old_pick(*old,scan,bridge.categories)
        k=previous["selected"]["new_full"]
        expected=(*k[:3],bridge.categories[k[3]],bridge.categories[k[4]])
        assert actual==expected,(actual,expected)
        after=one_slot(row["base"],actual,ci)
        assert torch.equal(after["label"],previous["labels"]["new_full"])
        assert all(after[k] is row["base"][k] for k in ("box","query","score"))
        assert one_slot(row["base"],None,ci) is row["base"]
        items.extend((a,row["nodes"],row["local"]) for a in row["actions"][:4])
    return items[:8],len(chosen)


def smoke(cfg):
    spatial_checks(cfg)
    label_checks()
    execution_checks()
    doc,rows=annotated_split("monitor")
    bridge=NativeBridge(doc,cfg)
    items,replayed=replay_checks(doc,bridge,cfg)
    assert len(items)>=4
    detail=model_checks(cfg,doc,items,bridge)
    sample=next(r for r in rows if r["annotations"])
    row,scan=bridge.scan(safe_frame(sample))
    records=annotate(sample,doc,row,scan["pack"]["boxes"],
                     {a["key"]:a["donor_query_ids"] for a in row["actions"]},store_matches=True)
    assert len(records)==len(row["actions"])
    prior=read(HISTORY/"smoke.json")
    assert prior["max_native16_error"]==0 and prior["max_risk_error"]==0
    report=dict(spatial=True,supervision=True,math=True,gradients=True,frozen=True,one_slot=True,
                history_parity=replayed,old_native16_max_error=prior["max_native16_error"],
                old_risk_max_error=prior["max_risk_error"],local=bridge.local.identity,detail=detail,
                code={p.name:digest(p) for p in CODE.glob("*.py")})
    save_json(OUT/"smoke.json",report)
    print("Smoke",report,flush=True)
