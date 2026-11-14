"""Training and evaluator labels. This module is outside the prediction path."""
from collections import Counter
import math

import numpy as np
import torch

from counterfactual import ImageReplay
from omni_layer_cache import adjudicate
from graphllm_local_message_unique_v2.core import semantic_label

SUPPORT, CONTRAST, INVALID = 0,1,2
STATUS = ("known_harm","unknown","safe_new","correct_no_new","verified_other")


def relation_label(receiver,donor):
    if receiver["owner"] is not None and donor["owner"]==receiver["owner"] and donor["clear"]:
        return INVALID
    if receiver["owner"] is not None and donor["source"]=="contradicted":
        return INVALID
    if receiver["owner"] is not None and donor["owner"] is not None and donor["source"]=="valid":
        if receiver["semantic"] in (0,1,2):
            return SUPPORT if receiver["semantic"]==1 else CONTRAST
    return -1


def label_utility(semantic, legacy, new, lost):
    harm=bool(legacy["wrong"] or any(lost))
    confirm=semantic==1
    reason=None
    if harm:
        gain=0
    elif legacy["target"]<0:
        gain,reason=-1,"legacy_utility_unknown"
    elif semantic<0:
        gain,reason=-1,"v2_semantic_unverified"
    elif bool(legacy["correct"])!=confirm:
        gain,reason=-1,"legacy_v2_claim_conflict"
    else:
        gain=int(confirm and (bool(new[0]) or bool(new[3])))
    result=dict(semantic_target=semantic,confirmed_correct=confirm,legacy_utility_known=legacy["target"]>=0,
                legacy_confirmed_correct=bool(legacy["correct"]),utility_known=gain>=0,
                original_class_harmed=bool(legacy["wrong"]),known_harm=harm,gain_target=gain,
                new=[len(v) for v in new],lost=[len(v) for v in lost],utility_reason=reason)
    result["state"]=status(result)
    return result


def status(t):
    if t["known_harm"]:
        return STATUS[0]
    if not t["utility_known"]:
        return STATUS[1]
    if t["gain_target"]==1:
        return STATUS[2]
    return STATUS[3] if t["confirmed_correct"] else STATUS[4]


def annotate(frame,doc,row,boxes,actions,store_matches=False,preserve_slot=True):
    if not actions:
        return {}
    ids=doc["categories"]
    by_index={ids[s]:i for i,s in enumerate(doc["base"])}
    # Keep the original Objects365 coverage flags and crowd replay. Ignored GT do not own a box.
    frame=dict(frame,annotations=[a for a in frame["annotations"] if not a.get("ignore",0)])
    known=set(frame["known"])
    replay=ImageReplay(frame,dict(boxes=boxes),row["base"],doc,doc["names"])
    regular=[a["id"] for a in frame["annotations"] if not a["iscrowd"] and a["category_id"] in by_index]
    assert len(regular)==len(replay.y)
    klist=list(actions)
    candidates=torch.tensor([(k[1],by_index[k[4]],k[2]) for k in klist])
    old=adjudicate(frame,dict(boxes=boxes),row["base"],doc,candidates) if not preserve_slot else None
    memo={}

    def inspect(q,c0,c1,is_receiver):
        key=q,c0,c1,is_receiver
        if key in memo:
            return memo[key]
        a,b=by_index[c0],by_index[c1]
        ious=replay.iou[q]
        labels=replay.y
        related=(a,b) if is_receiver else (b,)
        clear=not bool(((replay.ioa[q]>=.50)&np.isin(replay.cy,related)).any())
        cover=all(c in known for c in ((c0,c1) if is_receiver else (c1,)))
        hits=np.flatnonzero(ious>=.50)
        top=int(ious.argmax()) if len(ious) and ious.max()>=.50 else None
        u1=float(ious[labels==b].max(initial=0))
        owner=None
        if clear and cover and len(hits)==1 and ids[doc["base"][int(labels[top])]] in known:
            owner=int(regular[top])
        y=-1
        if is_receiver and clear and cover:
            y=semantic_label(torch.from_numpy(ious),torch.from_numpy(labels),a,b,True)
            if top is not None and ids[doc["base"][int(labels[top])]] not in known:
                y=-1
        source="unknown"
        if owner is not None:
            if labels[top]==b:
                source="valid"
            elif u1<=.30:
                source="contradicted"
        result=dict(owner=owner,semantic=y,source=source,clear=clear,cover=cover)
        memo[key]=result
        return result

    def matches(key):
        _,q,slot,c0,c1=key
        if not all(c in known for c in (c0,c1)):
            return [set() for _ in range(6)],[set() for _ in range(6)]
        a,b=by_index[c0],by_index[c1]
        if ((replay.ioa[q]>=.50)&np.isin(replay.cy,[a,b])).any():
            return [set() for _ in range(6)],[set() for _ in range(6)]
        if not preserve_slot:
            new,lost,*_=replay.delta(("replace",q,b,slot),float(replay.bs[slot]))
            return new,lost
        # Same greedy matcher, preserving original stable tie order because v2 edits in place.
        new=[set() for _ in range(6)]
        lost=[set() for _ in range(6)]
        for c in (a,b):
            slots=replay.by_class.get(c,np.empty(0,dtype=int))
            slots=slots[slots!=slot] if c==a else np.sort(np.append(slots,slot))
            after,_=replay.match_class(c,replay.bq[slots],replay.bs[slots])
            before=replay.before.get(c,replay.empty)
            for j,((p,_),(n,_)) in enumerate(zip(before,after)):
                new[j]|=n-p
                lost[j]|=p-n
        return new,lost

    result={}
    for i,key in enumerate(klist):
        _,q,_,c0,c1=key
        info=inspect(q,c0,c1,True)
        new,lost=matches(key)
        if old is None:
            a,b=by_index[c0],by_index[c1]
            support=[float(replay.iou[q,replay.y==c].max(initial=0)) for c in (a,b)]
            aligned=float(replay.iou[q].max(initial=0))>=.50
            verified=info["cover"] and info["clear"]
            # Legacy verification used a replacement appended for score ties. Replay it only when needed.
            proposed_slots=replay.by_class.get(b,np.empty(0,dtype=int))
            tied_later=(replay.bs[proposed_slots]==replay.bs[key[2]])&(proposed_slots>key[2])
            o=adjudicate(frame,dict(boxes=boxes),row["base"],doc,candidates[i:i+1])[0] if (
                verified and tied_later.any()) else None
            if o is None:
                wrong=verified and support[0]>=.50 and support[1]<.50
                target=-1
                if verified:
                    if wrong or any(lost) or (aligned and support[1]<=.30):
                        target=0
                    elif support[0]<.50 and support[1]>=.50:
                        target=1
                o=dict(wrong=wrong,target=target,correct=target==1,lost=[len(v) for v in lost])
        else:
            o=old[i]
        if not preserve_slot:
            assert [len(v) for v in new]==o["new"] and [len(v) for v in lost]==o["lost"]
        t=label_utility(info["semantic"],o,new,lost)
        if info["semantic"]>=0:
            # A verified semantic c1 requires the same class IoU condition as the legacy verifier.
            if info["semantic"]==1 and o["target"]>=0 and not any(o["lost"]):
                assert o["correct"],key
            if o["correct"]:
                assert info["semantic"] not in (0,2),key
        t.update(owner_gt_id=info["owner"],owner_known=info["owner"] is not None,
                 coverage_verified=info["cover"],crowd_clear=info["clear"])
        t["relation_target"]=[relation_label(info,inspect(d,c0,c1,False)) for d in actions[key]]
        if store_matches:
            t["new_ids"]=[[regular[i] for i in sorted(x)] for x in new]
            t["lost_ids"]=[[regular[i] for i in sorted(x)] for x in lost]
        assert not t["new"][2] and not t["new"][5] and not t["lost"][2] and not t["lost"][5],key
        result[key]=t
    return result


def masks(y,relation,edge):
    known=(relation>=0)&edge
    good=known&(relation!=INVALID)
    bad=known&(relation==INVALID)
    all_bad=edge.any(1)&(bad|~edge).all(1)
    semantic=(y>=0)&(known|~edge).all(1)&good.any(1)
    assert not (semantic&all_bad).any()
    return semantic,known,bad,all_bad


def weights(atoms, required):
    """Balance state, image, original action, then view/edge."""
    groups={}
    for s,key in atoms:
        groups.setdefault(s,{}).setdefault(key[0],Counter())[key]+=1
    if required is not None:
        assert set(groups)==set(required),{"required":required,"observed":list(groups)}
    out=[]
    for s,k in atoms:
        g=groups[s]
        count=g[k[0]]
        out.append(1/(len(groups)*len(g)*len(count)*count[k]))
    assert not out or math.isclose(sum(out),1.,abs_tol=1e-9)
    return out


def policy_prefix(rows,unknown_budget):
    from itertools import groupby

    score=sorted(rows,key=lambda r:(-r[0],r[1]))
    best=dict(gain_logit_cutoff=None,actions=0,correct=0,unknown=0,harm=0,new=[0]*6,lost=[0]*6)
    obj=(0,0,0,0)
    n=correct=unknown=harm=0
    new,lost=[0]*6,[0]*6
    for cutoff,group in groupby(score,key=lambda r:r[0]):
        assert math.isfinite(cutoff)
        for _,_,t in group:
            n+=1
            correct+=int(t["confirmed_correct"])
            unknown+=int(not t["utility_known"])
            harm+=int(t["known_harm"])
            new=[x+y for x,y in zip(new,t["new"])]
            lost=[x+y for x,y in zip(lost,t["lost"])]
        if harm or any(lost) or unknown>unknown_budget:
            break
        value=(new[0]+new[3],correct,-unknown,-n)
        if value>obj:
            best=dict(gain_logit_cutoff=cutoff,actions=n,correct=correct,unknown=unknown,harm=harm,new=new,lost=lost)
            obj=value
    return best
