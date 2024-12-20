import torch
import numpy as np
import random
from math import log10
import torch.nn as nn
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class AvgMeter(object):
    def __init__(self, num=40):
        self.num = num
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.losses.append(val)

    def show(self):
        return torch.mean(torch.stack(self.losses[np.maximum(len(self.losses)-self.num, 0):]))
