"""Trace receiver identity through frozen GraphLLM tokens on verified pairs."""
import argparse
from collections import Counter
from pathlib import Path

import torch
from PIL import Image

from layer_entropy_cache import encode, make_base
from objects365_cache import WideGraphPack
from objects365_data import FOLDER, INDEX, SHARDS, digest, read
from objects365_fit import qualified
from objects365_receiver_pair_views import PLAN as PAIR_PLAN, blocks, edge_rows

OUT = FOLDER/"receiver_token_probe_stock"
PLAN = Path(__file__).with_name("OBJECTS365_RECEIVER_TOKEN_PROBE.md")


def states(result,position):
    final = result.last_hidden_state[:,position].float()
    earlier = torch.stack([x[:,position].float() for x in result.hidden_states[1:-1]]).mean(0)
    return torch.cat((final,earlier),dim=-1)


@torch.inference_mode()
def capture(reader, pack, base, actions):
    model = reader.graph.star
    roles = torch.column_stack((base["label"][actions[:,2]],actions[:,1])).flatten()
    output = []
    start = 0

    def hook(module,args,kwargs,result):
        nonlocal start
        seq = kwargs["inputs_embeds"]
        n = len(seq)
        text = model.texts[roles[start:start+n].to(seq.device)]
        start += n
        own = states(result,0)[1::2]
        current = states(result,-1).reshape(-1,3584)
        masked_args = dict(kwargs)
        attention = kwargs["attention_mask"].clone()
        attention[:,1] = 0
        masked_args["attention_mask"] = attention
        masked = module.forward(**masked_args)
        anchored_args = dict(kwargs)
        anchored = seq.clone()
        anchored[:,2] = (seq[:,0].float()+text).to(seq.dtype)
        anchored_args["inputs_embeds"] = anchored
        revised = states(module.forward(**anchored_args),-1)
        output.append(dict(projection=seq[1::2,0].float().cpu(),receiver_token=own.cpu(),
                           current_proposal=current[:,1792:].cpu(),current_both=current.cpu(),
                           masked_proposal=states(masked,-1)[1::2].cpu(),
                           anchored_proposal=revised[1::2].cpu(),
                           anchored_both=revised.reshape(-1,3584).cpu()))

    handle = model.qwen.register_forward_hook(hook,with_kwargs=True)
    try:
        stock = encode(reader,pack,base,actions)[:,0]
    finally:
        handle.remove()
    out = {k:torch.cat([p[k] for p in output]).half() for k in output[0]}
    assert torch.equal(stock,out["current_both"])
    return out


@torch.inference_mode()
def trace(reader,engine,pack,base,record,source,head):
    device = reader.graph.star.texts.device
    actions = record["action"].flatten(0,1)
    q,c,s = actions.T
    assert torch.equal(base["query"][s],q)
    receivers,proposal_mask = edge_rows(pack,actions)
    actual = pack["node_ids"][pack["j"][receivers]]
    donor_id = record["donor"].repeat_interleave(2)
    selected = proposal_mask & (actual==donor_id[:,None])
    assert (selected.sum(1)==1).all()
    all_actions = source["actions"]
    lookup = {tuple(row):i for i,row in enumerate(all_actions.tolist())}
    ids = torch.tensor([lookup[tuple(row)] for row in actions.tolist()])
    pool = capture(reader,pack,base,all_actions)
    values = {k:v[ids].clone() for k,v in pool.items()}
    for edge in selected.long().argmax(1).unique().tolist():
        subset = (selected[:,edge])&(proposal_mask.sum(1)>1)
        if not subset.any():
            continue
        valid = pack["valid"].clone()
        mask = proposal_mask[subset].clone()
        mask[:,edge] = False
        valid[receivers[subset]] &= ~mask
        isolated = capture(reader,dict(pack,valid=valid),base,actions[subset])
        for key,value in isolated.items():
            values[key][subset] = value
    reference = record["x"].reshape(-1,3584).float().to(device)
    old_score = head(reference).softmax(-1)[:,1]
    new_score = head(values["current_both"].float().to(device)).softmax(-1)[:,1]
    gap = (old_score-new_score).abs().max().item()
    original = pack["features"][pack["i"][receivers]].to(device).float()
    chosen_donor = pack["features"][pack["j"][receivers][selected]].to(device).float()
    dino = pack["logits"][q,c]-pack["logits"][q,base["label"][s]]
    clip_text = engine.text[c.to(device)]-engine.text[base["label"][s].to(device)]
    clip_margin = (original[:,256:]*clip_text).sum(1)
    metrics = torch.column_stack((dino.to(device),clip_margin,new_score,
        torch.nn.functional.cosine_similarity(original[:,:256],chosen_donor[:,:256]),
        torch.nn.functional.cosine_similarity(original[:,256:],chosen_donor[:,256:])))
    n = len(record["donor"])
    values["raw"] = original.half().cpu()
    payload = {k:v.reshape(n,2,-1) for k,v in values.items()}
    assert all(torch.isfinite(v).all() for v in payload.values())
    return dict(action=record["action"],donor=record["donor"],old=base["label"][s].reshape(n,2),
                embedding=payload,metrics=metrics.reshape(n,2,-1).float().cpu(),gap=gap,
                scores=old_score.reshape(n,2).cpu(),gaps=(old_score-new_score).abs().reshape(n,2).cpu())



def run(doc,shard):
    engine = WideGraphPack(doc["names"],365,doc["parents"])
    head,receipt = qualified()
    for split,frames,prior,pairs,source_lock in blocks(doc,("train","rare_train","monitor"),shard):
        dest = OUT/pairs.name
        lock = dict(plan=digest(PLAN),code=digest(Path(__file__)),pairs=digest(pairs),
                    source=digest(prior),checkpoint=receipt["checkpoint"])
        if dest.exists():
            assert torch.load(dest,map_location="cpu",weights_only=True)["lock"]==lock
            continue
        saved = torch.load(pairs,map_location="cpu",weights_only=True)
        assert saved["lock"]["data"]==source_lock and saved["lock"]["source"]==lock["source"]
        assert saved["lock"]["plan"]==digest(PAIR_PLAN)
        refs = {r["image"]:r for r in torch.load(prior,map_location="cpu",weights_only=True)["records"]}
        out,counts = {},Counter()
        for frame in frames:
            iid = frame["id"]
            if iid not in saved["records"]:
                continue
            with Image.open(frame["path"]) as image:
                pack,_,_ = engine.native_pack(image.convert("RGB"))
            base = make_base(pack,365)
            value = trace(engine.reader,engine,pack,base,saved["records"][iid],refs[iid],head)
            out[iid] = value
            counts.update(images=1,pairs=len(value["donor"]))
            counts["parity_ppm"] = max(counts["parity_ppm"],round(value["gap"]*1e6))
        assert out.keys()==saved["records"].keys()
        temp = dest.with_suffix(".part")
        torch.save(dict(lock=lock,records=out,counts=dict(counts)),temp)
        temp.replace(dest)
        if out:
            print(dest.name,dict(counts),flush=True)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard",type=int,choices=range(SHARDS),required=True)
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    run(read(INDEX),args.shard)



