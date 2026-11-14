"""Blind, locked evaluation of the receiver keep gate on fresh Objects365 images."""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from layer_entropy_cache import encode, make_base
from objects365_cache import WideGraphPack
from objects365_claim_cache import ReceiverVisual
from objects365_coverage_cache import coverage
from objects365_data import BLOCK, INDEX, SHARDS, digest, read, save_json
from objects365_donor_view_entropy import OUT as GRAPH, paired_donors
from objects365_donor_view_entropy_eval import entropy, measure
from objects365_fit import qualified as original_head, replay_metrics
from objects365_instance_gate import ReceiverAttention
from objects365_receiver_keep_gate import OUT as KEEP, evidence, features, qualified
from objects365_receiver_keep_holdout_data import COUNT, NEW, OUT, PLAN
from objects365_receiver_native_evidence import OUT as NATIVE
from omni_layer_cache import adjudicate, eligible_actions

ARMS = ("old_entropy","keep_mean","keep_entropy")


def lock():
    return dict(plan=digest(PLAN),code=digest(Path(__file__)),policy=digest(KEEP/"policy.json"),
                keep=digest(KEEP/"qualification.json"),old=digest(GRAPH/"policy.json"),
                native=digest(NATIVE/"qualification.json"))


def chosen(score, risk, js, multiple, cap, js_cap, threshold, reselect):
    if threshold is None:
        return None
    use = risk<=cap
    if js_cap is not None:
        use &= (~multiple)|(js<=js_cap)
    order = score.masked_fill(~use,-torch.inf) if reselect else score
    j = int(order.argmax())
    return j if use[j] and float(order[j])>=threshold else None


class Engine:
    @torch.inference_mode()
    def __init__(self,doc):
        self.lock = lock()
        self.graph = WideGraphPack(doc["names"],365,doc["parents"])
        self.model,original = original_head()
        self.risk,risk = qualified()
        self.original = read(GRAPH/"policy.json")
        assert self.original["lock"]["checkpoint"]==original["checkpoint"]
        self.settings = read(KEEP/"policy.json")
        assert self.settings["qualification"]==self.lock["keep"] and risk["passed"]
        self.receiver = read(NATIVE/"qualification.json")
        assert risk["anchor"]["native"]==self.lock["native"]
        self.visual = ReceiverVisual.__new__(ReceiverVisual)
        self.visual.native = self.graph.native
        self.visual.attention = ReceiverAttention(self.visual)
        self.visual.clip = self.graph.reader.clip
        self.visual.processor = self.graph.reader.processor
        p = self.visual.processor(text=[f"a photo of a {name}." for name in doc["names"][:365]],
                                  padding=True,return_tensors="pt").to("cuda")
        self.visual.text = F.normalize(self.visual.clip.get_text_features(**p).float(),dim=-1)

    @torch.inference_mode()
    def infer(self,frame):
        with Image.open(frame["path"]) as file:
            image = file.convert("RGB")
        pack,_,_ = self.graph.native_pack(image)
        base = make_base(pack,365)
        raw = eligible_actions(pack,base,365)
        result = dict(base=base,boxes=pack["boxes"],picks={})
        if not raw:
            return result
        actions = torch.tensor(raw)[:,[0,1,3]]
        q,c,s = actions.T
        pool = encode(self.graph.reader,pack,base,actions)[:,0].float().cuda()
        p = self.model(pool).softmax(-1)[:,1].cpu()[:,None].repeat(1,2)
        idx,rows,edges,_,_ = paired_donors(pack,actions)
        for m in range(2):
            if not len(idx):
                break
            valid = pack["valid"].clone()
            valid[rows,edges[:,1-m]] = False
            x = encode(self.graph.reader,dict(pack,valid=valid),base,actions[idx])[:,0]
            p[idx,m] = self.model(x.float().cuda()).softmax(-1)[:,1].cpu()
        multiple = torch.zeros(len(q),dtype=torch.bool)
        multiple[idx] = True
        p = p.double()
        raw_mean = p.mean(1)
        raw_js = (entropy(raw_mean)-entropy(p).mean(1)).clamp_min(0)
        no_risk = torch.zeros_like(raw_mean)
        old_point = self.original["points"]["mixed_entropy"]["0"]
        picks = dict(old_entropy=chosen(raw_mean,no_risk,raw_js,multiple,1.,self.original["entropy_cap"],
                                        old_point["cutoff"],False))
        old = base["label"][s]
        margin = pack["logits"][q,c]-pack["logits"][q,old]
        settings = self.receiver
        z = (margin.double()-settings["center"])/settings["scale"]
        p = (p.clamp(1e-7,1-1e-7).logit()+settings["best_weight"]*z[:,None]).sigmoid()
        mean = p.mean(1)
        js = (entropy(mean)-entropy(p).mean(1)).clamp_min(0)
        proof = coverage(pack,base,actions)
        x = self.visual.views(image,pack["boxes"],actions,torch.column_stack((old,c)),proof)
        risk = self.risk(features(x.float())).sigmoid()
        for name,arm in (("keep_mean","mean"),("keep_entropy","mixed_entropy")):
            setting = self.settings["points"][arm]["limited"]
            picks[name] = chosen(mean,risk,js,multiple,setting["risk_cap"],setting["js_cap"],
                                 setting["point"]["cutoff"],True)
        result.update(actions=actions,picks={k:j for k,j in picks.items() if j is not None},
                      mean=mean,risk=risk,js=js,multiple=multiple,old_mean=raw_mean)
        assert all(result[k].isfinite().all() for k in ("mean","risk","js","old_mean"))
        return result


def parity():
    doc = read(INDEX)
    engine = Engine(doc)
    old = read(GRAPH/"result.json")["accepted"]["mixed_entropy"]["0"]
    expected = read(KEEP/"result.json")
    keep = expected["selected"]["mixed_entropy_limited"]
    cases = sorted({r["image"] for r in keep}|{r["image"] for r in old}|{4793,3494})
    cache,rows,_ = evidence(doc,"eval",engine.risk)
    base = {r["image"]:r for r in rows}
    arms = dict(old_entropy={r["image"]:r for r in old},
                keep_mean={r["image"]:r for r in expected["selected"]["mean_limited"]},
                keep_entropy={r["image"]:r for r in keep})
    max_score,max_risk,counts = 0.,0.,Counter()
    for iid in cases:
        if iid not in base:
            continue
        item = engine.infer(doc["rows"][iid])
        ref = base[iid]
        assert torch.equal(ref["boxes"],item["boxes"])
        assert all(torch.equal(ref["base"][k],item["base"][k]) for k in item["base"])
        if "actions" not in ref:
            assert "actions" not in item
            continue
        assert torch.equal(ref["actions"],item["actions"])
        max_score = max(max_score,float((item["mean"]-cache[iid]["mean"]).abs().max()))
        max_risk = max(max_risk,float((item["risk"]-cache[iid]["risk"]).abs().max()))
        for arm,selected in arms.items():
            r = selected.get(iid)
            index = item["picks"].get(arm)
            if r is None:
                assert index is None,(iid,arm,index)
            else:
                assert index is not None and item["actions"][index].tolist()==[r["query"],r["proposed"],r["slot"]],(iid,arm)
            counts["decisions"] += 1
        counts.update(images=1,actions=len(item["actions"]))
    assert max_score<=.002 and max_risk<=.002,(max_score,max_risk)
    result = dict(lock=engine.lock,counts=dict(counts),max_score=max_score,max_risk=max_risk,
                  peak_reserved_mib=int(torch.cuda.max_memory_reserved()/2**20))
    save_json(OUT/"parity.json",result)
    print("Parity",result,flush=True)


def blocks(doc,shard):
    rows = doc["rows"][shard::SHARDS]
    for part,start in enumerate(range(0,len(rows),BLOCK)):
        yield OUT/f"pred_{shard}_{part:03d}.pt",rows[start:start+BLOCK]


def block_lock(locked,frames):
    return dict(model=locked,parity=digest(OUT/"parity.json"),manifest=digest(NEW),images=[r["id"] for r in frames])


def infer(doc,shard):
    engine = Engine(doc)
    assert read(OUT/"parity.json")["lock"]==engine.lock
    assert doc["lock"]["policy"]==engine.lock["policy"]
    for path,frames in blocks(doc,shard):
        anchor = block_lock(engine.lock,frames)
        if path.exists():
            assert torch.load(path,map_location="cpu",weights_only=True)["lock"]==anchor
            continue
        records = {}
        for frame in frames:
            assert digest(Path(frame["path"]))==frame["sha"]
            blind = {k:frame[k] for k in ("id","path","file_name","width","height")}
            records[frame["id"]] = engine.infer(blind)
        temp = path.with_suffix(".part")
        torch.save(dict(lock=anchor,records=records),temp)
        temp.replace(path)
        print(path.name,len(records),sum(len(r["picks"]) for r in records.values()),flush=True)


def evaluate(doc):
    locked = lock()
    assert read(OUT/"parity.json")["lock"]==locked and locked["policy"]==doc["lock"]["policy"]
    rows,blocks_hash = {},[]
    for shard in range(SHARDS):
        for path,frames in blocks(doc,shard):
            b = torch.load(path,map_location="cpu",weights_only=True)
            assert b["lock"]==block_lock(locked,frames)
            assert set(b["records"])=={r["id"] for r in frames} and not (rows.keys()&b["records"].keys())
            rows.update(b["records"])
            blocks_hash.append(digest(path))
    assert len(rows)==len(doc["rows"])==COUNT
    selected = {k:[] for k in ARMS}
    baseline = []
    counts = Counter(images=len(rows))
    for iid,r in sorted(rows.items()):
        baseline.append(dict(image=iid,base=r["base"],boxes=r["boxes"]))
        if "actions" not in r:
            assert not r["picks"]
            continue
        actions = r["actions"]
        outcomes = adjudicate(doc["rows"][iid],r,r["base"],doc,actions)
        counts.update(candidate_images=1,actions=len(actions),possible=int(any(o["correct"] for o in outcomes)),
                      gain_possible=int(any(o["correct"] and (o["new"][0] or o["new"][3]) for o in outcomes)))
        for arm,j in r["picks"].items():
            q,c,s = actions[j].tolist()
            target = outcomes[j]
            selected[arm].append(dict(target,image=iid,query=q,proposed=c,slot=s,
                                      harmful=bool(target["wrong"] or any(target["lost"])),
                                      risk=float(r["risk"][j]),score=float(r["mean"][j]),
                                      js=float(r["js"][j]) if bool(r["multiple"][j]) else None))
    actual = {k:measure(v,counts["possible"],counts["candidate_images"]) for k,v in selected.items()}
    ap,classes = replay_metrics(doc,baseline,selected)
    generator = np.random.default_rng(20261112)
    draw = generator.integers(0,len(rows),(3000,len(rows)))
    paired = {}
    for name in ("keep_mean","keep_entropy"):
        values = np.zeros((len(rows),4),dtype=int)
        for sign,arm in ((-1,"old_entropy"),(1,name)):
            for r in selected[arm]:
                values[r["image"]] += sign*np.array([r["correct"],r["harmful"],r["target"]<0,r["new"][0]-r["lost"][0]])
        samples = values[draw].sum(1)
        paired[name] = {k:dict(delta=int(v.sum()),interval=np.quantile(s,(.025,.975)).tolist())
                        for k,v,s in zip(("correct","harmful","unknown","net_iou50"),values.T,samples.T)}
    result = dict(lock=locked,manifest=digest(NEW),blocks=blocks_hash,counts=dict(counts),ap=ap,
                  actual=actual,selected=selected,paired=paired,classes=classes)
    save_json(OUT/"result.json",result)
    print("Holdout",dict(counts),actual,paired,ap,flush=True)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase",choices=("parity","infer","evaluate"))
    parser.add_argument("--shard",type=int,choices=range(SHARDS))
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.phase=="parity":
        parity()
    elif args.phase=="infer":
        infer(read(NEW),args.shard)
    else:
        evaluate(read(NEW))

