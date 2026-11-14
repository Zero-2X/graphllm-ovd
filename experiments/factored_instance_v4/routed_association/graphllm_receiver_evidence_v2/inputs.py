"""Label-free tensor assembly shared by training and prediction."""
import torch


def batch(items,category_index,cfg,device):
    s=cfg["dimensions"]
    count=len(items)
    d=s["grid"]**2
    slots=cfg["candidate"]["max_edges"]
    edge_mask=torch.zeros(count,slots,dtype=torch.bool)
    node=torch.zeros(count,slots,s["node"],dtype=torch.float16)
    local=torch.zeros(count,slots,d,s["local"],dtype=torch.float16)
    edges=torch.zeros(count,slots,s["edge"],dtype=torch.float32)
    for i,(a,n,l) in enumerate(items):
        k=len(a["donors"])
        assert k<=slots
        if k:
            edge_mask[i,:k]=True
            node[i,:k]=n[a["donors"]]
            local[i,:k]=l[a["donors"]]
            edges[i,:k]=a["edge"]
    result=dict(
        classes=torch.tensor([[category_index[c] for c in a["key"][3:]] for a,_,_ in items]),
        receiver=torch.stack([n[a["receiver"]] for a,n,_ in items]),
        local=torch.stack([l[a["receiver"]] for a,_,l in items]),
        native=torch.stack([a["native"] for a,_,_ in items]),
        native16=torch.stack([a["native16"] for a,_,_ in items]),
        donor=node,donor_local=local,edge=edges,edge_mask=edge_mask,
    )
    return {k:v.to(device,non_blocking=True) for k,v in result.items()}


