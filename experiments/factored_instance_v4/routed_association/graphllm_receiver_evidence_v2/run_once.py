"""Ordered entry point for one receiver evidence run."""
import argparse

import torch

from .common import CODE, OUT, cache_write, checked, config, digest, feature_lock, init_data, read, save_json


def seal():
    from .data import load_dataset, supervision
    from .targets import weights

    assert not (OUT/"BUILD_COMPLETE.json").exists()
    cfg=config()
    lock=feature_lock()
    smoke=read(OUT/"smoke.json")
    assert smoke["code"]=={p.name:digest(p) for p in CODE.glob("*.py")}
    cache,supply={},{}
    for split in ("train","monitor"):
        data,labels,meta=load_dataset(split,verify=False)
        count,total,square=meta.pop("norm")
        if split=="train":
            assert count>0
            mean=total/count
            std=(square/count-mean.square()).clamp_min(0).sqrt().clamp_min(cfg["normalizer"]["floor"])
            cache_write(OUT/"native_stats.pt",dict(lock=lock,values=(mean.float(),std.float()),actions=count))
        assert smoke["local"]==meta["local"]
        cache[split]=meta
        supply[split]={stage:supervision(data,labels,stage,cfg)[1] for stage in ("self","message")}
        g=[(t["gain_target"],key) for (key,view),t in labels.items() if view==0 and t["gain_target"]>=0]
        weights(g,(0,1))
        supply[split]["gain"]={s:sum(y==s for y,_ in g) for s in (0,1)}
        del data,labels
    receipt=dict(feature_lock=lock,code={p.name:digest(p) for p in sorted(CODE.glob("*.py"))},
                 config=digest(CODE/"config.json"),data=digest(OUT/"data_lock.json"),
                 cache=cache,supply=supply,smoke=digest(OUT/"smoke.json"),local=smoke["local"],
                 native_stats=digest(OUT/"native_stats.pt"),
                 scope="Inputs, model, losses, training, policy, blind prediction and evaluator")
    save_json(OUT/"BUILD_COMPLETE.json",receipt)
    print("BUILD_COMPLETE",supply,flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=("init","smoke","cache","seal","self","message","gain",
                                       "policy-scores","lock","predict","seal-predictions","evaluate"))
    parser.add_argument("--split",choices=("train","monitor"))
    parser.add_argument("--shard",type=int)
    args=parser.parse_args()
    cfg=config()
    torch.set_num_threads(3)
    OUT.mkdir(exist_ok=True)
    if args.stage=="init":
        print({k:v["n"] for k,v in init_data()["sets"].items()},flush=True)
    elif args.stage=="smoke":
        from .smoke import smoke
        feature_lock()
        smoke(cfg)
    elif args.stage=="cache":
        from .data import cache
        assert args.split is not None and args.shard in range(cfg["workers"])
        assert (OUT/"smoke.json").is_file()
        cache(args.split,args.shard,cfg)
    elif args.stage=="seal":
        seal()
    elif args.stage in ("self","message"):
        from .train import train_graph
        train_graph(cfg,args.stage)
    elif args.stage=="gain":
        from .train import train_gain
        train_gain(cfg)
    elif args.stage=="lock":
        from .evaluate import lock_policy, policy_scores
        if not any(OUT.glob("policy_scores_*_*.pt")):
            for shard in range(cfg["workers"]):
                policy_scores(cfg,shard)
        lock_policy(cfg)
    elif args.stage=="policy-scores":
        from .evaluate import policy_scores
        assert args.shard in range(cfg["workers"])
        policy_scores(cfg,args.shard)
    elif args.stage=="predict":
        from .predict import predict
        assert args.shard in range(cfg["workers"])
        predict(cfg,args.shard)
    elif args.stage=="seal-predictions":
        from .predict import seal_predictions
        checked(args.stage)
        seal_predictions(cfg)
    else:
        from .evaluate import evaluate
        checked(args.stage)
        evaluate(cfg)


if __name__=="__main__":
    main()
