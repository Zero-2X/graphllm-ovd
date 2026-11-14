"""Action identity, shared three-token reader, and fixed one-slot policy."""
import math
from itertools import groupby

import torch
from torch import nn
from torch.nn import functional as F


KEY_FIELDS = ("image", "query", "slot", "original", "proposed")
SCHEMA = "local-message-v1/node768/native5/edge12/legacy-native16/gain18"
NATIVE5 = ("old_dino_logit", "new_dino_logit", "old_clip_fit", "new_clip_fit", "receiver_stability")
EDGE12 = ("clip_cos", "donor_score", "donor_clip_fit", "donor_stability", "aspect_difference", "parent_flag",
          "ioa_receiver", "ioa_donor", "relative_center_x", "relative_center_y", "relative_log_width", "relative_log_height")
EDGE_COLUMNS = (0, 2, 4, 5, 6, 7)


def iou(a, b):
    intersect = (torch.minimum(a[..., 2:], b[..., 2:]) - torch.maximum(a[..., :2], b[..., :2])).clamp_min(0).prod(-1)
    aa = (a[..., 2:]-a[..., :2]).prod(-1)
    ab = (b[..., 2:]-b[..., :2]).prod(-1)
    return intersect / (aa+ab-intersect).clamp_min(1e-12)


def edge_geometry(a, b):
    ra, rb = a[2:]-a[:2], b[2:]-b[:2]
    assert (ra>0).all() and (rb>0).all()
    inter = (torch.minimum(a[2:], b[2:])-torch.maximum(a[:2], b[:2])).clamp_min(0).prod()
    ioa = (inter/torch.stack((ra.prod(),rb.prod())).clamp_min(1e-6)).clamp(0,1)
    centers = ((b[:2]+b[2:])-(a[:2]+a[2:]))*.5 / ra.clamp_min(1e-6)
    ratios = (rb.clamp_min(1e-6)/ra.clamp_min(1e-6)).log()
    return torch.cat((ioa, centers.clamp(-3,3), ratios.clamp(-3,3)))


def build_actions(image, pack, base, raw, native16):
    """Consume legacy edges in their existing order and keep legacy feature parity."""
    owners = {int(q): i for i,q in reversed(list(enumerate(base["query"].tolist())))}
    existing = set(zip(base["query"].tolist(),base["label"].tolist()))
    rows = {q:i for i,q in enumerate(pack["ids"].tolist())}
    good, rejected = [], []
    for (q,c,slot),x in zip(raw.tolist(),native16):
        old = int(base["label"][slot])
        key = (image,q,slot,old,c)
        reason = None
        if (q,c) in existing:
            reason = "duplicate_slot"
        elif q not in rows or owners.get(q)!=slot or old==c or not float(base["score"][slot])<.35:
            reason = "invalid_input"
        if reason:
            rejected.append(dict(key=key,reject_reason=reason))
            continue
        r = rows[q]
        ri = int(pack["i"][r])
        receiver = pack["features"][ri]
        box = pack["boxes"][q]
        native = torch.cat((pack["logits"][q,[old,c]],pack["fit"][ri,[old,c]],pack["stability"][ri:ri+1]))
        if not all(t.isfinite().all() for t in (native,receiver,box,x)) or not (box[2:]>box[:2]).all():
            rejected.append(dict(key=key,reject_reason="invalid_input"))
            continue
        donors, edges, queries, quality = [],[],[],[]
        for e in range(min(2,pack["valid"].shape[1])):
            if not pack["valid"][r,e] or int(pack["labels"][r,e])!=c:
                continue
            di = int(pack["j"][r,e])
            dq = int(pack["node_ids"][di])
            db = pack["boxes"][dq]
            if dq==q or dq in queries or not db.isfinite().all() or not (db[2:]>db[:2]).all():
                continue
            if any(iou(db,pack["boxes"][p])>=.50 for p in queries):
                continue
            donor = pack["features"][di]
            attrs = torch.cat((pack["stats_full"][r,e,list(EDGE_COLUMNS)],edge_geometry(box,db)))
            assert int(pack["logits"][dq,:365].argmax())==c
            if not attrs.isfinite().all() or not donor.isfinite().all():
                rejected.append(dict(key=key,reject_reason="invalid_input"))
                break
            queries.append(dq)
            donors.append(donor)
            edges.append(attrs)
            quality.append((attrs[1]*attrs[3]).clamp(0,1))
        else:
            count = len(queries)
            good.append(dict(key=key,receiver=receiver,native5=native,native16=x,
                             donors=torch.stack(donors) if count else receiver.new_empty((0,768)),
                             edges=torch.stack(edges) if count else native.new_empty((0,12)),
                             rho=torch.stack(quality) if count else native.new_empty(0),
                             donor_query_ids=queries,donor_predicted_classes=[c]*count))
    return good,rejected


def semantic_label(ious, labels, c0, c1, verifiable):
    if not verifiable or not len(ious) or not ious.max()>=.50:
        return -1
    maximum = ious.max()
    owners = labels[ious==maximum].unique()
    if len(owners)!=1:
        return -1
    u0 = ious[labels==c0].max() if (labels==c0).any() else 0.
    u1 = ious[labels==c1].max() if (labels==c1).any() else 0.
    owner = int(owners[0])
    if owner==c0 and u0>=.50 and u1<=.30:
        return 0
    if owner==c1 and u1>=.50 and u0<=.30:
        return 1
    if owner not in (c0,c1) and u0<=.30 and u1<=.30:
        return 2
    return -1


def gain_label(outcome):
    if outcome["target"]<0:
        return -1
    return int(bool(outcome["correct"] and not any(outcome["lost"]) and
                    (outcome["new"][0]>0 or outcome["new"][3]>0)))


def fuse(z_self, z_donor, rho, cfg, training, lam=None):
    z_self,z_donor,rho = z_self.float(),z_donor.float(),rho.float()
    v = z_donor-z_self[:,None]
    center = v-v.mean(-1,keepdim=True)
    delta = center*(cfg["bound"]/(center.abs().amax(-1,keepdim=True)+cfg["epsilon"])).clamp(max=1)
    valid = rho>0
    delta = delta.masked_fill(~valid[...,None],0)
    mass = rho.sum(1,keepdim=True)
    weights = rho/torch.where(mass>0,mass,torch.ones_like(mass))
    quality = (weights*rho).sum(1)
    q = (delta/cfg["temperature"]).softmax(-1)
    def entropy(p):
        return -(p*p.clamp_min(cfg["epsilon"]).log()).sum(-1)
    js = (entropy((weights[...,None]*q).sum(1))-(weights*entropy(q)).sum(1)).clamp_min(0)
    js_available = valid.sum(1)>=2
    js = torch.where(js_available,js,0.)
    gate = (js.detach() if training else js)/math.log(3)
    beta = quality*torch.exp(-(cfg["lambda"] if lam is None else lam)*gate)
    fused = z_self+beta[:,None]*(weights[...,None]*delta).sum(1)
    return dict(fused=fused,delta=delta,js=js,beta=beta,quality_abs=quality,weights=weights,
                js_available=js_available,has_valid_donor=valid.any(1))


def pack_actions(rows,device):
    inputs = {k:torch.stack([r[k] for r in rows]).to(device)
              for k in ("receiver","native5","native16")}
    for k,shape in (("donors",(2,768)),("edges",(2,12)),("rho",(2,))):
        value = torch.zeros((len(rows),*shape),dtype=rows[0][k].dtype)
        for i,row in enumerate(rows):
            value[i,:len(row[k])] = row[k]
        inputs[k] = value.to(device)
    inputs["classes"] = torch.tensor([r["key"][3:] for r in rows],device=device)
    return inputs


class Reader(nn.Module):
    def __init__(self, star, dtype):
        super().__init__()
        self.qwen = star.qwen.to(dtype=dtype).requires_grad_(False).eval()
        self.register_buffer("texts",star.texts.detach().clone(),persistent=False)
        self.embedding_scale = star.embedding_scale
        self.node = nn.Sequential(nn.LayerNorm(768),nn.Linear(768,896))
        self.native = nn.Linear(5,896)
        self.edge = nn.Linear(12,896)
        self.receiver_role = nn.Parameter(torch.zeros(896))
        self.donor_role = nn.Parameter(torch.zeros(896))
        self.null = nn.Parameter(torch.empty(896))
        self.read = nn.Parameter(torch.empty(896))
        self.head = nn.Sequential(nn.LayerNorm(3584),nn.Linear(3584,768),nn.GELU(),nn.Linear(768,256),nn.GELU(),nn.Linear(256,3))
        self.register_buffer("native_mean",torch.zeros(5))
        self.register_buffer("native_std",torch.ones(5))
        self.migrate(star)

    @torch.no_grad()
    def migrate(self,star):
        self.node.load_state_dict(star.instance.state_dict())
        self.read.copy_(star.read)
        assert star.edge.weight.shape==(896,8) and (star.edge.bias is not None)
        self.edge.weight.zero_()
        self.edge.weight[:,:6].copy_(star.edge.weight[:,list(EDGE_COLUMNS)])
        self.edge.bias.copy_(star.edge.bias)
        self.native.weight.zero_()
        self.native.bias.zero_()
        self.null.normal_(0,self.embedding_scale)

    def train(self,mode=True):
        super().train(mode)
        self.qwen.eval()
        return self

    def tokens(self,b):
        ns = ((b["native5"]-self.native_mean)/self.native_std).clamp(-6,6)
        receiver = self.embedding_scale*(self.node(b["receiver"].float())+self.native(ns))+self.receiver_role
        donor = self.embedding_scale*(self.node(b["donors"].float())+self.edge(b["edges"].float()))
        donor = donor+self.texts[b["classes"][:,1],None]+self.donor_role
        donor = torch.cat(((self.null+self.donor_role).expand(len(receiver),1,-1),donor),dim=1)
        read = self.read+self.texts[b["classes"]]
        return receiver,donor,read

    def forward(self,b):
        receiver,donor,read = self.tokens(b)
        valid = torch.cat((torch.ones_like(b["rho"][:,:1],dtype=torch.bool),b["rho"]>0),dim=1)
        ix,view = valid.nonzero(as_tuple=True)
        a = receiver[ix,None,None].expand(-1,2,1,-1)
        d = donor[ix,view,None,None].expand(-1,2,1,-1)
        tokens = torch.cat((a,d,read[ix,:,None]),dim=2).flatten(0,1).to(self.qwen.dtype)
        pos = torch.arange(3,device=tokens.device).expand(len(tokens),-1)
        out = self.qwen(inputs_embeds=tokens,attention_mask=torch.ones_like(pos),position_ids=pos,
                        output_hidden_states=True,use_cache=False)
        last = out.last_hidden_state[:,-1].float()
        early = torch.stack([h[:,-1].float() for h in out.hidden_states[1:-1]]).mean(0)
        logits = self.head(torch.cat((last,early),dim=-1).reshape(len(ix),3584)).float()
        all_logits = logits.new_zeros((len(receiver),3,3)).index_put((ix,view),logits)
        return all_logits[:,0],all_logits[:,1:]

    def artifact(self):
        return {k:v.detach().cpu() for k,v in self.state_dict().items() if not k.startswith("qwen.")}

    def restore(self,state):
        current = self.state_dict()
        assert state.keys()=={k for k in current if not k.startswith("qwen.")}
        current.update(state)
        self.load_state_dict(current,strict=True)


class Gain(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("mean",torch.zeros(18))
        self.register_buffer("std",torch.ones(18))
        self.layer = nn.Linear(18,1)
        nn.init.zeros_(self.layer.weight)
        nn.init.zeros_(self.layer.bias)

    def forward(self,x):
        return self.layer(((x-self.mean)/self.std).clamp(-6,6)).flatten()


def gain_features(x,z):
    return torch.cat((x,(z[:,1]-z[:,0])[:,None],z.softmax(-1)[:,2:3]),dim=1)


def semantic_loss(z,d,f,rho,y):
    self_loss = F.cross_entropy(z,y,reduction="none")
    donor_loss = F.cross_entropy(d.flatten(0,1),y.repeat_interleave(d.shape[1]),reduction="none").reshape_as(rho)
    donor_loss = (donor_loss*(rho>0)).sum(1)/(rho>0).sum(1).clamp_min(1)
    return F.cross_entropy(f,y,reduction="none")+.25*self_loss+.25*donor_loss


def select(records):
    eligible = [r for r in records if r.get("reject_reason") is None]
    if not eligible:
        return None
    return min(eligible,key=lambda r:(-r["gain_score"],tuple(r["key"])))


def apply(base,key, reject_duplicate=True):
    if key is None:
        return base
    _,q,s,old,new = key
    assert int(base["query"][s])==q and int(base["label"][s])==old and old!=new
    assert float(base["score"][s])<.35
    if reject_duplicate:
        assert not ((base["query"]==q)&(base["label"]==new)).any()
    changed = base["label"].clone()
    changed[s] = new
    assert int((changed!=base["label"]).sum())==1
    return dict(base,label=changed)


def choose_prefix(winners, unknown_budget):
    ordered = sorted(winners,key=lambda r:-r["gain_score"])
    best_key = (0,0,0,0)
    best = dict(gain_cutoff=None,actions=0,correct=0,unknown=0,new=[0,0],harm=0,lost=[0]*6)
    actions=correct=unknown=harm=0
    new,lost = [0,0],[0]*6
    for score, group in groupby(ordered,key=lambda r:r["gain_score"]):
        for r in group:
            y = r["target"]
            actions+=1
            correct+=y["correct"]
            unknown+=int(y["target"]<0)
            harm+=int(bool(y["wrong"] or any(y["lost"])))
            new=[a+y["new"][i] for a,i in zip(new,(0,3))]
            lost=[a+b for a,b in zip(lost,y["lost"])]
        if harm or any(lost) or unknown>unknown_budget:
            break
        objective = (sum(new),correct,-unknown,-actions)
        if objective>best_key:
            best_key=objective
            best=dict(gain_cutoff=score,actions=actions,correct=correct,unknown=unknown,new=new,harm=harm,lost=lost)
    return best
