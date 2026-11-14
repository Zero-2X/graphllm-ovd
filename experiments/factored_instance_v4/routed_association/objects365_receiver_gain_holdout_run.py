"""Compare two fixed gain targets on fresh Objects365 images."""
import argparse
import gc
from pathlib import Path

import numpy as np
import torch

import objects365_receiver_gain_gate as gain
import objects365_receiver_keep_holdout_data as data
import objects365_receiver_keep_holdout_run as holdout
from objects365_data import FOLDER, INDEX, SHARDS, digest, read, save_json


OUT = FOLDER/"receiver_gain_holdout"
NEW = OUT/"manifest.json"
PLAN = Path(__file__).with_name("OBJECTS365_RECEIVER_GAIN_HOLDOUT.md")
COUNT = 2048
PARENT_LOCK, PARENT_ENGINE = holdout.lock, holdout.Engine
HEADS = {name:FOLDER/f"receiver_{name}_gate" for name in ("gain","gain50")}


def lock():
    state = dict(PARENT_LOCK(), extension=digest(Path(__file__)), sampler=digest(Path(data.__file__)))
    for arm, folder in HEADS.items():
        state[arm] = {name:digest(folder/f"{name}.json") for name in ("qualification","policy","result")}
    return state


def load_head(name, folder, model_lock):
    receipt, policy, result = [read(folder/f"{key}.json") for key in ("qualification","policy","result")]
    assert receipt["passed"] and receipt["anchor"]["keep_policy"]==model_lock["policy"]
    assert receipt["anchor"]["code"]==digest(Path(gain.__file__))
    if name=="gain50":
        assert receipt["anchor"]["target_code"]==digest(Path(__file__).with_name("objects365_receiver_gain50_gate.py"))
    assert policy["qualification"]==model_lock[name]["qualification"]
    assert result["policy"]==model_lock[name]["policy"]
    assert receipt["best_hash"]==digest(folder/"best.pt")
    state = torch.load(folder/"best.pt",map_location="cpu",weights_only=True)
    assert state["anchor"]==receipt["anchor"] and state["step"]==receipt["best_step"]
    model = gain.Gain().eval()
    model.load_state_dict(state["model"])
    return model,policy["points"]["limited"]["cutoff"]


class Engine(PARENT_ENGINE):
    def __init__(self,doc):
        super().__init__(doc)
        self.heads = {name:load_head(name,folder,self.lock) for name,folder in HEADS.items()}
        self.model.register_forward_hook(self.capture_graph)
        self.risk.register_forward_pre_hook(self.capture_claim)

    def capture_graph(self, _module, _args, output):
        # The parent reads the pooled original crop before any donor-specific views.
        if self.graph_logit is None:
            self.graph_logit = (output[:,1]-output[:,0]).cpu()

    def capture_claim(self, _module, args):
        self.claim = args[0]

    @torch.inference_mode()
    def infer(self,frame):
        self.graph_logit, self.claim = None,None
        row = super().infer(frame)
        if "actions" not in row:
            assert self.graph_logit is None and self.claim is None
            return row
        assert self.claim.shape==(len(row["actions"]),16) and self.graph_logit.shape==(len(row["actions"]),)
        x = torch.cat((self.claim,self.graph_logit[:,None]),1)
        cfg = self.settings["points"]["mixed_entropy"]["limited"]
        row["gain_scores"] = {}
        for arm, (head, cutoff) in self.heads.items():
            score = head(x).sigmoid().double()
            row["gain_scores"][arm] = score
            choice = holdout.chosen(.5+.5*score,row["risk"],row["js"],row["multiple"],
                                    cfg["risk_cap"],cfg["js_cap"],cutoff,True)
            if choice is not None:
                row["picks"][arm] = choice
        return row


def parity():
    holdout.parity()
    receipt = read(OUT/"parity.json")
    gc.collect()
    torch.cuda.empty_cache()
    doc = read(INDEX)
    engine = Engine(doc)
    expected = {name:{r["image"]:r for r in read(path/"result.json")["selected"]["limited"]}
                for name,path in HEADS.items()}
    images = sorted({i for r in expected.values() for i in r})
    maxima = {arm:0. for arm in HEADS}
    for iid in images:
        row = engine.infer(doc["rows"][iid])
        for arm,wanted in expected.items():
            selected = row["picks"].get(arm)
            if iid not in wanted:
                assert selected is None, (iid,arm,selected)
                continue
            r = wanted[iid]
            assert selected is not None and row["actions"][selected].tolist()==[r["query"],r["proposed"],r["slot"]], (iid,arm)
            maxima[arm] = max(maxima[arm],abs(float(row["gain_scores"][arm][selected])-r["score"]))
    assert max(maxima.values())<=.002,maxima
    receipt.update(gain_parity=dict(images=len(images),decisions=len(images)*len(HEADS),max_difference=maxima),
                   peak_reserved_mib=max(receipt["peak_reserved_mib"],int(torch.cuda.max_memory_reserved()/2**20)))
    save_json(OUT/"parity.json",receipt)
    print("Gain parity",receipt["gain_parity"],flush=True)


def evaluate():
    holdout.evaluate(read(NEW))
    path = OUT/"result.json"
    result = read(path)
    n = result["counts"]["images"]
    draws = np.random.default_rng(20261112).integers(0,n,(3000,n))
    comparisons = {}
    for after in HEADS:
        values = np.zeros((n,6),dtype=int)
        for sign,arm in ((-1,"keep_entropy"),(1,after)):
            for row in result["selected"][arm]:
                values[row["image"]] += sign*np.array([row["correct"],row["target"]<0,row["harmful"],
                    row["new"][0]-row["lost"][0],row["new"][3]-row["lost"][3],sum(row["lost"])])
        samples = values[draws].sum(1)
        comparisons[after] = {key:dict(delta=int(col.sum()),interval=np.quantile(sample,(.025,.975)).tolist())
            for key,col,sample in zip(("correct","unknown","harmful","net50","net75","loss_six"),values.T,samples.T)}
    result["gain_paired"] = comparisons
    save_json(path,result)
    print("Gain comparison",comparisons,flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase",choices=("prepare","parity","infer","evaluate"))
    parser.add_argument("--shard",type=int,choices=range(SHARDS))
    args = parser.parse_args()
    torch.set_num_threads(4)
    for module in (data,holdout):
        module.OUT,module.NEW,module.PLAN,module.COUNT = OUT,NEW,PLAN,COUNT
    holdout.Engine,holdout.lock = Engine,lock
    holdout.ARMS = (*holdout.ARMS,*HEADS)
    if args.phase=="prepare":
        lock()
        data.prepare()
    elif args.phase=="parity":
        parity()
    elif args.phase=="infer":
        assert "gain_parity" in read(OUT/"parity.json")
        holdout.infer(read(NEW),args.shard)
    else:
        evaluate()


if __name__ == "__main__":
    main()
