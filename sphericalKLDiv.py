import numpy as np
import torch
import torch.cuda
import torch.nn as nn
from matplotlib import pyplot as plt
from numba import jit
from torch.autograd import Function
from numba import cuda
import math

class KLWeightedLossSequence(nn.Module):
    def __init__(self):
        super(KLWeightedLossSequence, self).__init__()
        self.epsilon = 1e-8  # the parameter to make sure the denominator non-zero

    def forward(self, map_pred, map_gtd):  # map_pred : input prediction saliency map, map_gtd : input ground truth density map

        bs, nf, H, W = map_pred.shape
        map_pred = map_pred.float()
        map_gtd = map_gtd.float()

        map_pred = map_pred.view(bs, nf, -1)  # change the map_pred into a tensor with n rows and 1 cols
        map_gtd = map_gtd.view(bs, nf, -1)  # change the map_pred into a tensor with n rows and 1 cols

        min1, _ = torch.min(map_pred, dim=2, keepdim=True)
        max1, _ = torch.max(map_pred, dim=2, keepdim=True)

        map_pred = (map_pred - min1) / (max1 - min1 + self.epsilon)  # min-max normalization for keeping KL loss non-NAN

        min2, _ = torch.min(map_gtd, dim=2, keepdim=True)
        max2, _ = torch.max(map_gtd, dim=2, keepdim=True)

        map_gtd = (map_gtd - min2) / (max2 - min2 + self.epsilon)  # min-max normalization for keeping KL loss non-NAN

        map_pred = map_pred / (
                    torch.sum(map_pred, dim=2, keepdim=True) + self.epsilon)  # normalization step to make sure that the map_pred sum to 1
        map_gtd = map_gtd / (
                    torch.sum(map_gtd, dim=2, keepdim=True) + self.epsilon)  # normalization step to make sure that the map_gtd sum to 1

        # Calculate the weights
        weight = np.zeros((H, W))
        theta_range = np.linspace(0, np.pi, num=H + 1)
        dtheta = np.pi / H
        dphi = 2 * np.pi / W
        for theta_idx in range(H):
            weight[theta_idx, :] = dphi * (
                    np.sin(theta_range[theta_idx]) + np.sin(theta_range[theta_idx + 1])) / 2 * dtheta

        weight = torch.Tensor(weight).unsqueeze(0).unsqueeze(0).repeat(bs, nf, 1, 1).view(bs, nf, -1).to(map_pred.device)

        KL = torch.log(map_gtd / (map_pred + self.epsilon) + self.epsilon)
        KL = weight * map_gtd * KL
        KL = torch.sum(KL, dim=2)

        return KL