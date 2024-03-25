"""
Implementation for the Memory Bank for pixel-level feature vectors
"""

import torch
import numpy as np
import random

from torch.autograd import Function
from torch import nn
import math
import torch.nn.functional as F

class LinearAverageOp(Function):
    @staticmethod
    def forward(self, x, memory):
        #T = params[0].item()
        batchSize = x.size(0)
        x_flat = x.view(batchSize,-1)
        x_flat = F.normalize(x_flat, dim=1)

        # inner product
        #out = torch.mm(x.data, memory.t())
        #out.div_(T) # batchSize * N

        memory = memory * 0.05 + (1-0.05) * x_flat.data
        memory = F.normalize(memory, dim=1)

        self.save_for_backward(x, memory)

        return memory

    @staticmethod
    def backward(self, gradOutput):
        x, memory = self.saved_tensors
        batchSize = gradOutput.size(0)
        # T = params[0].item()
        # momentum = params[1].item()

        # add temperature
        # gradOutput.data.div_(T)

        # gradient of linear
        #gradInput = torch.mm(gradOutput.data, memory
        gradInput = gradOutput
        gradInput.resize_as_(x)

        # update the non-parametric data
        #weight_pos = memory.index_select(0, y.data.view(-1)).resize_as_(x)
        # weight_pos.mul_(momentum)
        # weight_pos.add_(torch.mul(x.data, 1-momentum))
        # w_norm = weight_pos.pow(2).sum(1, keepdim=True).pow(0.5)
        # updated_weight = weight_pos.div(w_norm)
        #memory.index_copy_(0, y, weight_pos)
        return gradInput, None, None

class LinearAverage(nn.Module):

    def __init__(self):
        super(LinearAverage, self).__init__()
        # stdv = 1 / math.sqrt(inputSize)
        # self.nLem = outputSize
        #
        # self.register_buffer('params',torch.tensor([T, momentum]));
        # stdv = 1. / math.sqrt(inputSize/3)
        self.register_buffer('memory', torch.ones(8, 393216))

    def forward(self, x):
        out = LinearAverageOp.apply(x, self.memory)
        return out

class FeatureMemory:
    def __init__(self, memory_size=2048, feature_size=256, n_classes=2):

        self.memory_size = memory_size
        self.feature_size = feature_size
        self.memory = None
        self.n_classes = n_classes

    def build_memory(self, predictionS, predictionT):

        batch_size = predictionS.size()[0]
        predictionS = predictionS.view(batch_size,-1)
        predictionT = predictionT.view(batch_size,-1)

        prediction = torch.cat((predictionS, predictionT), 0)

        #prediction = prediction.detach().cpu().numpy()
        #prediction = predictionT

        if self.memory is None: # was empy, first elements
            self.memory = prediction

        else: # add elements to already existing list
            # keep only most recent memory_per_class samples
            self.memory = torch.cat((prediction, self.memory), axis = 0)[:self.memory_size, :]
