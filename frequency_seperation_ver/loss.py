import torch
from torch import nn
import torch.nn.functional as F


class SCLoss(nn.Module):
    def __init__(self, lmbda, delta):
        super(SCLoss, self).__init__()
        self.lmbda = lmbda
        self.delta = delta
    def forward(self, input, target, latent):
        assert input.shape==target.shape
        loss_l2 = 0.5*torch.norm(target-input)**2
        loss_huber = F.huber_loss(input,target,delta=self.delta)
        loss_norm = torch.norm(latent)
        losses = {'Reconstruction':loss_huber, 'Regularization':loss_norm}
        return loss_huber+self.lmbda*loss_norm, losses