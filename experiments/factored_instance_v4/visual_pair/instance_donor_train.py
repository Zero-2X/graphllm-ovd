"""Train a bounded correspondence head with one variant per labeled instance group."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from learned_donor_train import DonorMatcher, SEED, WIDTH, main

NAME = "instance_donor"
BOUND = 3.
EPOCHS = 48


class CosineEvidence(nn.Module):
    def __init__(self):
        super().__init__()
        self.direction = nn.Parameter(torch.empty(1,WIDTH))
        nn.init.normal_(self.direction,std=WIDTH**-.5)

    def forward(self,x):
        direction = F.normalize(self.direction,dim=-1)
        return BOUND*F.linear(F.normalize(x,dim=-1),direction)


class InstanceDonorMatcher(DonorMatcher):
    def __init__(self):
        super().__init__()
        self.extent[-1] = CosineEvidence()
        self.match[-1] = CosineEvidence()


class InstanceSampler:
    def __init__(self, rows, device="cuda"):
        groups, inverse, counts = np.unique(rows[:,[4,2,5,3]],axis=0,return_inverse=True,return_counts=True)
        rng = np.random.default_rng(SEED)
        shuffled = rng.permutation(len(rows))
        self.variants = shuffled[np.argsort(inverse[shuffled],kind="stable")]
        self.starts = np.r_[0,np.cumsum(counts[:-1])]
        self.counts = counts
        self.device = device
        self.rng = rng
        _, class_index, class_count = np.unique(groups[:,:2],axis=0,return_inverse=True,return_counts=True)
        weight = class_count[class_index].astype(float)**-.5
        weight /= np.bincount(groups[:,0],weights=weight)[groups[:,0]]
        self.weights = weight/weight.mean()
        self.summary = dict(groups=len(groups),states=np.bincount(groups[:,0],minlength=3).tolist(),
                            largest_group=int(counts.max()),max_weight=float(self.weights.max()),
                            effective=float(self.weights.sum()**2/(self.weights**2).sum()))
        print("instance sampler",self.summary,flush=True)

    def epoch(self, epoch):
        perm = self.rng.permutation(len(self.counts))
        index = self.variants[self.starts+(epoch-1)%self.counts][perm]
        return (torch.from_numpy(index).to(self.device),
                torch.from_numpy(self.weights[perm]).float().to(self.device))


if __name__=="__main__":
    main(InstanceDonorMatcher,NAME,"INSTANCE_DONOR_PLAN_ZH.md",EPOCHS,InstanceSampler)
