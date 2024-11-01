import numpy as np
import torch
from .strategy import Strategy

# Use the prediction entropy as uncertainty
class EntropySampling(Strategy):
    def __init__(self, dataset, net):
        super(EntropySampling, self).__init__(dataset, net)

    def query(self, n, index,param2,param3):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data(index = index)
        probs = self.predict_prob(unlabeled_data) #([12384, 1, 128, 128])
        log_probs = torch.log(probs)#([12384, 1, 128, 128])
        uncertainties = (probs*log_probs).sum((1,2,3))#([12384])
        return unlabeled_idxs[uncertainties.sort()[1][:n]]


