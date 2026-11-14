"""Eight bounded correctness checks, including real legacy parity."""
import hashlib

import torch

from .bridge import FOLDER, OUT, Vision, data_sources, digest, load, read, read_model, safe_frame, save_json
from .core import (EDGE_COLUMNS, Gain, apply, build_actions, choose_prefix, edge_geometry, fuse, gain_features,
                   gain_label, pack_actions, semantic_label, semantic_loss)


def close(x,y,tol=1e-6):
    torch.testing.assert_close(torch.as_tensor(x),torch.as_tensor(y),atol=tol,rtol=tol)


def synthetic(cfg):
    checks={}
    for ious,labels,verified,expected in (
        ([.50,.30],[0,1],True,0),([.30,.50],[0,1],True,1),([.15,.10,.5],[0,1,2],True,2),
        ([.61,.55],[0,1],True,-1),([.5,.31],[0,1],True,-1),([.56,.1],[0,1],False,-1),
        ([.3,.2],[0,1],True,-1),([.6,.6],[0,2],True,-1),([],[],True,-1),
        ([.5,.5],[1,1],True,1)):
        assert semantic_label(torch.tensor(ious),torch.tensor(labels),0,1,verified)==expected
    checks["labels"]=True
    z=torch.tensor([[3.,-1.,0.],[1.,0.,-.5],[2.,0.,1.]])
    d=torch.tensor([[[5.,-9.,1.],[-8.,7.,2.]],[[2.,-2.,1.],[2.,-2.,1.]],[[2.,0.,1.],[2.,0.,1.]]])
    rho=torch.tensor([[.6,.2],[.01,0.],[0.,0.]])
    a=fuse(z,d,rho,cfg["message"],False)
    close(a["delta"].sum(-1),torch.zeros(3,2))
    assert a["delta"].abs().max()<=2.000001 and (a["delta"]<0).any()
    close(a["fused"],fuse(z+1,d+3,rho,cfg["message"],False)["fused"]-1)
    close(a["delta"],fuse(z+1,d+3,rho,cfg["message"],False)["delta"])
    assert a["js_available"].tolist()==[True,False,False] and a["js"][1]==0 and a["js"][0]>0
    identical=fuse(z,d[:,0:1].expand(-1,2,-1),torch.ones(3,2),cfg["message"],False)
    close(identical["js"],torch.zeros(3))
    checks["residual_js"]=True
    close(a["fused"],fuse(z,d.flip(1),rho.flip(1),cfg["message"],False)["fused"])
    assert torch.equal(a["fused"][2],z[2])
    close(fuse(z,z[:,None].expand(-1,2,-1),rho,cfg["message"],False)["fused"],z)
    close(a["beta"][1],torch.tensor(.01))
    close(edge_geometry(torch.tensor([0.,0.,1.,1.]),torch.tensor([0.,0.,.5,.5])),
          torch.tensor([.25,1.,-.25,-.25,-.69314718,-.69314718]))
    box=torch.tensor([[0.,0.,.1,.1],[.3,.3,.4,.4],[.3,.3,.4,.4]])
    logits=torch.zeros(3,365)
    logits[1:,1]=1.
    pack=dict(boxes=box,logits=logits,ids=torch.tensor([0]),node_ids=torch.arange(3),i=torch.tensor([0]),
              j=torch.tensor([[1,2]]),valid=torch.ones((1,2),dtype=torch.bool),labels=torch.tensor([[1,1]]),
              features=torch.ones(3,768),fit=torch.zeros(3,365),stability=torch.ones(3),
              stats_full=torch.ones(1,2,8))
    base=dict(query=torch.tensor([0]),label=torch.tensor([0]),score=torch.tensor([.2]),box=box[:1])
    rows,_=build_actions(0,pack,base,torch.tensor([[0,1,0]]),torch.zeros(1,16))
    assert rows[0]["donor_query_ids"]==[1]
    pack["j"][0,1]=1
    rows,_=build_actions(0,pack,base,torch.tensor([[0,1,0]]),torch.zeros(1,16))
    assert rows[0]["donor_query_ids"]==[1]
    checks["fusion_fallback"]=True
    result=apply(base,(0,0,0,0,1))
    assert apply(base,None) is base and result["label"].tolist()==[1] and base["label"].tolist()==[0]
    assert all(torch.equal(base[k],result[k]) for k in ("query","score","box"))
    dupe=dict(query=base["query"].repeat(2),label=torch.tensor([0,1]),score=base["score"].repeat(2),box=base["box"].repeat(2,1))
    picked,rejected=build_actions(0,pack,dupe,torch.tensor([[0,1,0]]),torch.zeros(1,16))
    assert not picked and rejected[0]["reject_reason"]=="duplicate_slot"
    checks["slot_constraints"]=True
    native=torch.arange(48).reshape(3,16).float()
    x=gain_features(native,z)
    assert x.shape==(3,18) and torch.equal(x[:,:16],native)
    close(x[:,16],z[:,1]-z[:,0])
    close(x[:,17],z.softmax(1)[:,2])
    model=Gain()
    state=model.state_dict()
    try:
        model.load_state_dict(dict(state,mean=torch.zeros(17)),strict=True)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Accepted the old 17-dimensional head")
    y=dict(target=1,correct=1,wrong=0,lost=[0]*6,new=[1,0,0,0,0,0])
    assert gain_label(y)==1 and gain_label(dict(y,target=-1))==-1 and gain_label(dict(y,lost=[0]*5+[1]))==0
    tie=[dict(gain_score=.8,target=y),dict(gain_score=.8,target=dict(y,wrong=1))]
    assert choose_prefix(tie,11)["gain_cutoff"] is None
    checks["gain_interface"]=True
    off=fuse(z,d,rho,cfg["message"],False,0.)
    assert torch.equal(a["delta"],off["delta"]) and torch.equal(a["js"],off["js"]) and not torch.equal(a["fused"],off["fused"])
    checks["isolated_js_intervention"]=True
    return checks


def smoke(cfg):
    from objects365_claim_cache import OUT as CLAIM
    from objects365_data import BLOCK, SHARDS
    from objects365_receiver_keep_gate import features
    from selective import V1
    checks=synthetic(cfg)
    resolved=read(OUT/"resolved_config.json")
    doc,splits=data_sources()
    previous=read(FOLDER/"receiver_gain_gate/result.json")["selected"]["limited"]
    expected={r["image"]:r for r in previous}
    frames=[(i,r) for i,r in enumerate(splits["eval"]) if r["id"] in expected][:cfg["smoke_max"]]
    assert frames
    engine=Vision(doc,legacy=True)
    model=read_model(doc,cfg,resolved)
    disk=load(V1/"best.pt")
    close(model.edge.weight[:,:6].cpu(),disk["edge.weight"][:,list(EDGE_COLUMNS)])
    close(model.edge.bias.cpu(),disk["edge.bias"])
    close(model.node[1].weight.cpu(),disk["instance.1.weight"])
    close(model.read.cpu(),disk["read"])
    assert not model.edge.weight[:,6:].count_nonzero() and not model.native.weight.count_nonzero()
    all_frozen=tuple(p for m in (model.qwen,engine.graph.native.model,engine.visual.clip) for p in m.parameters())
    assert all(not p.requires_grad for p in all_frozen)
    def fingerprint():
        state=hashlib.sha256()
        for p in all_frozen:
            state.update(p.detach().reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
        return state.hexdigest()
    before=fingerprint()
    max_feature_error=max_risk_error=0.
    gradient_batch=None
    n=0
    for pos,frame in frames:
        safe=safe_frame(frame)
        assert "annotations" not in safe
        scan=engine.scan(safe)
        j=pos%SHARDS
        part=pos//(SHARDS*BLOCK)
        actual=engine.legacy_pick(safe,scan)
        want=expected[frame["id"]]
        assert actual is not None and (actual[1],actual[4],actual[2])==(want["query"],want["proposed"],want["slot"]),frame["id"]
        ref=load(CLAIM/f"eval_{j}_{part:03d}.pt")["features"][frame["id"]]
        by_key={tuple(r):i for i,r in enumerate(ref["actions"].tolist())}
        keys=scan["raw"].tolist()
        common=[(i,by_key[tuple(k)]) for i,k in enumerate(keys) if tuple(k) in by_key]
        assert common and any(keys[i]==[want["query"],want["proposed"],want["slot"]] for i,_ in common)
        a,b=torch.tensor(common).T
        x=scan["native16"][a]
        y=features(ref["x"][b].float())
        max_feature_error=max(max_feature_error,float((x-y).abs().max()))
        with torch.inference_mode():
            max_risk_error=max(max_risk_error,float((engine.risk(x).sigmoid()-engine.risk(y).sigmoid()).abs().max()))
        cached=load(FOLDER/f"eval_{j}_{part:03d}.pt")["records"]
        old=next(r for r in cached if r["image"]==frame["id"])["base"]
        assert all(torch.equal(scan["base"][k],old[k]) for k in old)
        if gradient_batch is None and any(len(r["rho"]) for r in scan["actions"]):
            gradient_batch=scan["actions"][:8]
        n+=len(common)
    assert max_feature_error<=.001 and max_risk_error<=.001,(max_feature_error,max_risk_error)
    assert gradient_batch and n>0
    checks["gain_risk_interfaces"]=checks.pop("gain_interface")
    b=pack_actions(gradient_batch,"cuda")
    r,donor,readout=model.tokens(b)
    assert r.shape==(len(b["rho"]),896) and donor.shape==(len(b["rho"]),3,896) and readout.shape==(len(b["rho"]),2,896)
    close(donor[:,0],(model.null+model.donor_role).expand_as(donor[:,0]))
    changed=dict(b,classes=b["classes"].clone())
    changed["classes"][:,0]=(changed["classes"][:,0]+1)%365
    r2,d2,read2=model.tokens(changed)
    assert torch.equal(r,r2) and torch.equal(donor,d2) and torch.equal(readout[:,1],read2[:,1])
    assert not torch.equal(readout[:,0],read2[:,0])
    def inspect(_module,_args,kw):
        length=len(kw["inputs_embeds"])
        assert kw["attention_mask"].tolist()==[[1,1,1]]*length
        assert kw["position_ids"].tolist()==[[0,1,2]]*length
    handle=model.qwen.register_forward_pre_hook(inspect,with_kwargs=True)
    model.train()
    assert not model.qwen.training
    z,d=model(b)
    handle.remove()
    assert z.shape==(len(b["rho"]),3) and d.shape==(len(b["rho"]),2,3)
    checks["read_context"]=True
    y=torch.arange(len(z),device="cuda")%3
    merged=fuse(z,d,b["rho"],cfg["message"],True)
    semantic_loss(z,d,merged["fused"],b["rho"],y).mean().backward()
    params={name:p for name,p in model.named_parameters() if p.requires_grad}
    assert all(p.grad is not None and p.grad.isfinite().all() for p in params.values())
    for block in ("node.","native.","edge.","head."):
        assert any(p.grad.count_nonzero() for k,p in params.items() if k.startswith(block)),block
    assert all(p.grad is None for p in all_frozen) and fingerprint()==before
    checks["frozen_gradients"]=True
    from .bridge import fresh_frame_digest
    blind=read(OUT/"holdout_images.json")
    assert fresh_frame_digest(blind)==resolved["blind_hash"]
    assert all(set(f)=={"id","source_id","file_name","path","width","height","sha"} for f in blind["rows"])
    assert fresh_frame_digest(dict(blind,rows=[]))!=resolved["blind_hash"]
    assert len(checks)==8
    save_json(OUT/"smoke.json",dict(resolved=digest(OUT/"resolved_config.json"),checks=checks,images=len(frames),
        native_action_pairs=n,max_native16_error=max_feature_error,max_risk_error=max_risk_error,
        trainable_count=sum(p.numel() for p in params.values()),trainable=list(params)))
    print("Smoke passed",checks,"images",len(frames),"native16",max_feature_error,"risk",max_risk_error,flush=True)
