"""Estimate category agreement across the local evidence of both boxes."""
import torch

from instance_donor_train import EPOCHS, InstanceDonorMatcher, InstanceSampler
from learned_donor_train import WIDTH, main

NAME = "patch_consensus"


class PatchConsensusMatcher(InstanceDonorMatcher):
    def patch_probability(self, query, reference, cue, similarity):
        weights = (similarity/.12).softmax(2)
        aligned = weights @ reference
        visual = query*aligned
        linguistic = (aligned*cue,query*cue)
        support = similarity.amax(2,keepdim=True)
        reciprocal = weights @ similarity.amax(1).unsqueeze(2)
        content = WIDTH*torch.cat((visual,*linguistic),dim=-1)
        return self.match(torch.cat((content,support,reciprocal),dim=-1)).squeeze(-1).sigmoid()

    def compare(self, source, receiver, cue):
        similarity = receiver @ source.transpose(1,2)
        cue = cue[:,None]
        to_receiver = self.patch_probability(receiver,source,cue,similarity)
        to_source = self.patch_probability(source,receiver,cue,similarity.transpose(1,2))
        same = (to_receiver.mean(1)+to_source.mean(1))*.5
        return self.extent_logit(receiver),torch.logit(same)


if __name__=="__main__":
    main(PatchConsensusMatcher,NAME,"PATCH_CONSENSUS_PLAN_ZH.md",EPOCHS,InstanceSampler)
