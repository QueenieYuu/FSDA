import torch
import torch.nn.functional as F
import torch.nn as nn
import math

from networks.models import GlobalDiscriminator, LocalDiscriminator, PriorDiscriminator

from torch import Tensor
from torch import einsum

from typing import Any, Callable, Iterable, List, Set, Tuple, TypeVar, Union


def entropy_loss(p, C=2):
    y1 = -1.0*torch.sum(p*torch.log(p+1e-6), dim=1)/torch.tensor(np.log(C)).cuda()
    ent = torch.mean(y1)

    return ent

def probs2one_hot(probs: Tensor) -> Tensor:
    _, C, _, _ = probs.shape
    # assert simplex(probs)

    res = class2one_hot(probs2class(probs), C)
    assert res.shape == probs.shape
    # assert one_hot(res)

    return res

def norm_soft_size(a: Tensor, power:int) -> Tensor:
    b, c, w, h = a.shape
    sl_sz = w*h
    amax = a.max(dim=1, keepdim=True)[0]+1e-10
    #amax = torch.cat(c*[amax], dim=1)
    resp = (torch.div(a,amax))**power
    ress = einsum("bcwh->bc", [resp]).type(torch.float32)
    ress_norm = ress/(torch.sum(ress,dim=1,keepdim=True)+1e-10)
    #print(torch.sum(ress,dim=1))
    return ress_norm.unsqueeze(2)

def class2one_hot(seg: Tensor, C: int) -> Tensor:
    if len(seg.shape) == 2:  # Only w, h, used by the dataloader
        seg = seg.unsqueeze(dim=0)
    #print('range classes:',list(range(C)))
    #print('unique seg:',torch.unique(seg))
    #print("setdiff:",set(torch.unique(seg)).difference(list(range(C))))
    # assert sset(seg, list(range(C)))

    b, w, h = seg.shape  # type: Tuple[int, int, int]

    res = torch.stack([seg == c for c in range(C)], dim=1).type(torch.int32)
    assert res.shape == (b, C, w, h)
    # assert one_hot(res)

    return res

def probs2class(probs: Tensor) -> Tensor:
    b, _, w, h = probs.shape  # type: Tuple[int, int, int, int]
    # assert simplex(probs)

    res = probs.argmax(dim=1)
    assert res.shape == (b, w, h)

    return res


class CrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean'):
        super().__init__(weight, size_average, ignore_index, reduce, reduction)

    def forward(self, logits: torch.tensor, target: torch.tensor, **kwargs):
        return super().forward(logits, target)


class StochasticSegmentationNetworkLossMCIntegral(nn.Module):
    def __init__(self, num_mc_samples: int = 1):
        super().__init__()
        self.num_mc_samples = num_mc_samples

    @staticmethod
    def fixed_re_parametrization_trick(dist, num_samples):
        assert num_samples % 2 == 0
        samples = dist.rsample((num_samples // 2,))
        mean = dist.mean.unsqueeze(0)
        samples = samples - mean
        return torch.cat([samples, -samples]) + mean

    def forward(self, logits, target, distribution, **kwargs):
        batch_size = logits.shape[0]
        num_classes = logits.shape[1]
        assert num_classes >= 2  # not implemented for binary case with implied background
        # logit_sample = distribution.rsample((self.num_mc_samples,))
        logit_sample = self.fixed_re_parametrization_trick(distribution, self.num_mc_samples)
        target = target.unsqueeze(1)
        target = target.expand((self.num_mc_samples,) + target.shape)

        flat_size = self.num_mc_samples * batch_size
        logit_sample = logit_sample.view((flat_size, num_classes, -1))
        target = target.reshape((flat_size, -1))

        # log_prob = -F.cross_entropy(logit_sample, target, reduction='none').view((self.num_mc_samples, batch_size, -1))
        log_prob = -F.binary_cross_entropy(F.sigmoid(logit_sample), target, reduction='none').view((self.num_mc_samples, batch_size, -1))
        loglikelihood = torch.mean(torch.logsumexp(torch.sum(log_prob, dim=-1), dim=0) - math.log(self.num_mc_samples))
        loss = -loglikelihood
        return loss


class DeepInfoMaxLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=1, gamma=0.1):
        super().__init__()
        self.global_d = GlobalDiscriminator()
        self.local_d = LocalDiscriminator()
        self.prior_d = PriorDiscriminator()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, M, M_prime):

        # see appendix 1A of https://arxiv.org/pdf/1808.06670.pdf

        M = M.unsqueeze(-1).unsqueeze(-1)
        M = M.expand(-1, -1, 128, 128)

        M_prime = M_prime.unsqueeze(-1).unsqueeze(-1)
        M_prime = M_prime.expand(-1, -1, 128, 128)
        #
        # y_M = torch.cat((M, y_exp), dim=1)
        # y_M_prime = torch.cat((M_prime, y_exp), dim=1)

        Ej = -F.softplus(-self.local_d(M)).mean()
        Em = F.softplus(self.local_d(M_prime)).mean()
        LOCAL = (Em - Ej) * self.beta

        # Ej = -F.softplus(-self.global_d(y, M)).mean()
        # Em = F.softplus(self.global_d(y, M_prime)).mean()
        # GLOBAL = (Em - Ej) * self.alpha
        #
        # prior = torch.rand_like(y)
        #
        # term_a = torch.log(self.prior_d(prior)).mean()
        # term_b = torch.log(1.0 - self.prior_d(y)).mean()
        # PRIOR = - (term_a + term_b) * self.gamma

        # print(LOCAL)
        # print(GLOBAL)
        # print(PRIOR)
        return LOCAL

class EntKLProp():
    """
    CE between proportions
    """
    def __init__(self, **kwargs):
        self.power = 1
        # self.__fn__ = getattr(__import__('utils'), kwargs['fn'])
        self.curi = True
        self.idc = [1]
        self.ivd = True
        self.weights = [0.1 ,0.9]
        self.lamb_se = 1
        self.lamb_conspred = 1
        self.lamb_consprior = 1

    def __call__(self, probs: Tensor, target: Tensor) -> Tensor:
        # assert simplex(probs)  # and simplex(target)  # Actually, does not care about second part
        b, _, w, h = probs.shape  # type: Tuple[int, int, int, int]
        predicted_mask = probs2one_hot(probs).detach()
        est_prop_mask = norm_soft_size(predicted_mask,self.power).squeeze(2)
        est_prop: Tensor = norm_soft_size(probs,self.power)
        if self.curi: # this is the case for the main experiments, i.e. we use curriculum learning. Put self.curi=True to reproduce the method
            # if self.ivd:
            #     bounds = bounds[:,:,0]
            #     bounds= bounds.unsqueeze(2)
            gt_prop = torch.ones_like(est_prop)*torch.rand(1,2,1).cuda()/(w*h)
            gt_prop = gt_prop[:,:,0]
        else: # for toy experiments, you can try and use the GT size calculated from the target instead of an estimation of the size.
            #Note that this is "cheating", meaning you are adding supplementary info. But interesting to obtain an upper bound
            gt_prop: Tensor = norm_soft_size(target,self.power) # the power here is actually useless if we have 0/1 gt labels
            gt_prop = gt_prop.squeeze(2)
        est_prop = est_prop.squeeze(2)
        log_est_prop: Tensor = abs(est_prop + 1e-10).log()

        log_gt_prop: Tensor = abs(gt_prop + 1e-10).log()
        log_est_prop_mask: Tensor = abs(est_prop_mask + 1e-10).log()

        loss_cons_prior = - torch.einsum("bc,bc->", [est_prop, log_gt_prop])  + torch.einsum("bc,bc->", [est_prop, log_est_prop])
        # Adding division by batch_size to normalise
        loss_cons_prior /= b
        log_p: Tensor = abs(probs + 1e-10).log()
        mask: Tensor = probs.type((torch.float32))
        mask_weighted = torch.einsum("bcwh,c->bcwh", [mask, Tensor(self.weights).to(mask.device)])
        loss_se = - torch.einsum("bcwh,bcwh->", [mask_weighted, log_p])
        loss_se /= mask.sum() + 1e-10

        assert loss_se.requires_grad == probs.requires_grad  # Handle the case for validation

        return self.lamb_se*loss_se, self.lamb_consprior*loss_cons_prior,est_prop
