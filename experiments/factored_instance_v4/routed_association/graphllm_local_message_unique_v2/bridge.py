"""Exact legacy detector, candidate and risk interfaces for v1."""
import hashlib
from pathlib import Path
from time import perf_counter

import torch
from PIL import Image
from torch.nn import functional as F
from torchvision.ops import box_iou,nms

from objects365_data import BASE, FOLDER, INDEX, SHARDS, digest, read, save_json
from objects365_rare import RARE_INDEX
from objects365_receiver_keep_gate import OUT as KEEP, features, qualified
from objects365_receiver_native_evidence import OUT as NATIVE
from objects365_donor_view_entropy import OUT as LEGACY_GRAPH
import objects365_receiver_gain_holdout_run as prior
from objects365_cache import WideGraphPack
from objects365_claim_cache import ReceiverVisual
from objects365_coverage_cache import coverage
from objects365_instance_gate import ReceiverAttention
from semantic_action_pool import eligible_actions
from layer_entropy_cache import make_base
from cache import GRAPH_BUDGET,build_graph,clip_crops,xyxy
from selective import V1

from .core import SCHEMA, Reader, build_actions, semantic_label, gain_label


OUT = FOLDER/"graphllm_local_message_unique_v2"
CODE = Path(__file__).resolve().parent
CONFIG = CODE/"config.json"
ADAPTER = V1/"best.pt"


def cache_write(path,value):
    temp = path.with_suffix(".part")
    torch.save(value,temp)
    temp.replace(path)


def load(path):
    return torch.load(path,map_location="cpu",weights_only=True)


def data_sources():
    main,rare = read(INDEX),read(RARE_INDEX)
    result = {s:[f for f in main["rows"] if f["split"]==s] for s in ("train","monitor","policy","eval")}
    result["train"] += rare["rows"]
    return main,result


def safe_frame(frame):
    return {k:frame[k] for k in ("id","source_id","file_name","path","width","height","sha")}


def fresh_frame_digest(doc):
    import json
    return hashlib.sha256(json.dumps(doc,sort_keys=True,ensure_ascii=False,separators=(",",":")).encode()).hexdigest()


def target_table(doc,frame,base,pack,actions):
    import numpy as np
    from counterfactual import ImageReplay
    from omni_layer_cache import adjudicate

    if not actions:
        return []
    replay = ImageReplay(frame,pack,base,doc,doc["names"])
    keys = [a["key"] for a in actions]
    raw = torch.tensor([(k[1],k[4],k[2]) for k in keys])
    utility = adjudicate(frame,pack,base,doc,raw)
    known = set(frame["known"])
    by_id = doc["categories"]
    output = []
    for key,out in zip(keys,utility):
        _,q,_,c0,c1 = key
        verifiable = (by_id[doc["names"][c0]] in known and by_id[doc["names"][c1]] in known
                      and not ((replay.ioa[q]>=.50)&np.isin(replay.cy,[c0,c1])).any())
        overlap = torch.from_numpy(replay.iou[q])
        labels = torch.from_numpy(replay.y)
        label = semantic_label(overlap,labels,c0,c1,verifiable)
        output.append(dict(key=key,semantic_target=label,utility=out,gain_target=gain_label(out)))
    return output


def read_model(doc,cfg,resolved=None):
    from model import StarAdapter
    torch.manual_seed(cfg["seed"])
    star = StarAdapter().cuda().eval()
    disk = load(ADAPTER)
    whole = star.state_dict()
    assert disk.keys()=={k for k in whole if not k.startswith("qwen.")}
    whole.update(disk)
    star.load_state_dict(whole,strict=True)
    star.requires_grad_(False)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    if resolved is not None:
        assert str(dtype)==resolved["dtype"]
    star.qwen.to(dtype=dtype)
    star.set_vocabulary(doc["names"],cfg["classes"],"cuda")
    return Reader(star,dtype).cuda()


class Vision:
    def __init__(self,doc,legacy=False):
        self.doc = doc
        self.legacy = None
        if legacy:
            prior.HEADS = {"gain":prior.HEADS["gain"]}
            prior.holdout.lock = prior.lock
            self.legacy = prior.Engine(doc)
            self.graph,self.visual,self.risk = self.legacy.graph,self.legacy.visual,self.legacy.risk
        else:
            self.graph = WideGraphPack(doc["names"],365,doc["parents"])
            self.risk,_ = qualified()
            self.visual = ReceiverVisual.__new__(ReceiverVisual)
            self.visual.native = self.graph.native
            self.visual.attention = ReceiverAttention(self.visual)
            self.visual.clip = self.graph.reader.clip
            self.visual.processor = self.graph.reader.processor
            with torch.inference_mode():
                text = self.visual.processor(text=[f"a photo of a {name}." for name in doc["names"][:365]],
                                             padding=True,return_tensors="pt").to("cuda")
                self.visual.text = F.normalize(self.visual.clip.get_text_features(**text).float(),dim=-1)
        self.graph.native.model.requires_grad_(False).eval()
        self.visual.clip.requires_grad_(False).eval()

    @torch.inference_mode()
    def scan(self,frame):
        path = Path(frame["path"])
        assert digest(path)==frame["sha"],path
        with Image.open(path) as source:
            image = source.convert("RGB")
        torch.cuda.synchronize()
        t = perf_counter()
        native = self.graph.native
        boxes,scores,field,scale = native.forward(image)
        pack = dict(boxes=boxes.cpu(),logits=scores.cpu())
        base = make_base(pack,365)
        torch.cuda.synchronize()
        native_time = perf_counter()-t
        t = perf_counter()
        positive = torch.where(((boxes[:,2:]-boxes[:,:2])>0).all(1))[0]
        nodes = positive[nms(boxes[positive],scores[positive].amax(-1),.82)[:GRAPH_BUDGET]]
        pack.update(ids=torch.empty(0,dtype=torch.long),node_ids=nodes.cpu())
        if len(nodes)>=2:
            subset = boxes[nodes]
            raw = native.last_output
            previous = xyxy(raw.intermediate_reference_points[0,-2])[nodes]
            stability = box_iou(subset,previous).diagonal()
            clip = clip_crops(image,subset,self.graph.reader.processor,self.graph.reader.clip)
            query = F.normalize(raw.last_hidden_state[0,nodes].float(),dim=-1)
            fit = clip@self.graph.text.T
            i,j,valid,labels,_,stats,high = build_graph(scores[nodes],subset,stability,clip,fit,self.graph.parents,365)
            pack.update(ids=nodes[i].cpu(),i=i.cpu(),j=j.cpu(),valid=valid.cpu(),labels=labels.cpu(),
                        stats=stats.half().cpu(),stats_full=stats.cpu(),high=high[i].cpu(),
                        features=torch.cat((query,clip),dim=1).half().cpu(),fit=fit.cpu(),stability=stability.cpu())
        raw_actions = eligible_actions(pack,base,365)
        raw = torch.tensor(raw_actions,dtype=torch.long).reshape(-1,4)[:,[0,1,3]]
        if len(raw):
            q,c,s = raw.T
            native8 = coverage(pack,base,raw)
            clip_native = self.visual.views(image,pack["boxes"],raw,torch.column_stack((base["label"][s],c)),native8)
            x = features(clip_native.float())
            actions,reject = build_actions(frame["id"],pack,base,raw,x)
        else:
            clip_native,x = None,torch.empty((0,16))
            actions,reject = [],[]
        torch.cuda.synchronize()
        feature_time = perf_counter()-t
        return dict(base=base,pack=pack,raw=raw,native16=x,raw_views=clip_native,actions=actions,rejected=reject,
                    field=field,scale=scale,native_time=native_time,feature_time=feature_time)

    @torch.inference_mode()
    def legacy_pick(self,frame,scan):
        if not len(scan["raw"]):
            return None
        actual_pack,actual_views = self.graph.native_pack,self.visual.views
        self.graph.native_pack = lambda image:(scan["pack"],scan["field"],scan["scale"])
        self.visual.views = lambda *_:scan["raw_views"]
        try:
            old = self.legacy.infer(frame)
        finally:
            self.graph.native_pack,self.visual.views = actual_pack,actual_views
        assert torch.equal(old["actions"],scan["raw"])
        assert torch.equal(self.legacy.claim,scan["native16"])
        j = old["picks"].get("gain")
        if j is None:
            return None
        q,c,s = old["actions"][j].tolist()
        return (frame["id"],q,s,int(scan["base"]["label"][s]),c)


def prepare_resolved(cfg, config_path):
    source = FOLDER/"graphllm_local_message_v1"
    source_resolved = read(source/"resolved_config.json")
    receipt = read(source/"build_receipt.json")
    assert receipt["complete"] and receipt["resolved"]==digest(source/"resolved_config.json")
    assert {k:v for k,v in cfg.items() if k!="version"}=={
        k:v for k,v in source_resolved["config"].items() if k!="version"}
    blind = read(source/"holdout_images.json")
    assert fresh_frame_digest(blind)==source_resolved["blind_hash"]
    for name in ("holdout_images.json","manifest.json"):
        if not (OUT/name).exists():
            (OUT/name).symlink_to(source/name)
    assert digest(OUT/"holdout_images.json")==digest(source/"holdout_images.json")
    original = str((source/"manifest.json").resolve())
    assert (OUT/"manifest.json").resolve()==Path(original)
    assert digest(OUT/"manifest.json")==source_resolved["source_hashes"][original]
    resolved = dict(source_resolved,config=cfg,config_hash=digest(config_path),
                    visual_source=receipt["resolved"],
                    code_hashes={p.name:digest(p) for p in sorted(CODE.glob("*.py"))})
    save_json(OUT/"resolved_config.json",resolved)
    return resolved
