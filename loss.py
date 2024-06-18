import torch
from torch import nn
import torch.nn.functional as F


class SCLoss(nn.Module):
    def __init__(self, lmbda, delta):
        super(SCLoss, self).__init__()
        self.lmbda = lmbda
        self.delta = delta
    def forward(self, input, target, latent, input_pca):
        assert input.shape==target.shape
        loss_l2 = 0.5*torch.norm(target-input)**2
        LR = F.huber_loss(input,target,delta=self.delta)
        LB = F.huber_loss(input_pca,target,delta=self.delta)
        loss_norm = torch.norm(latent)
        losses = {'Reconstruction':LR, 'Basis_Recon':LB, 'Regularization':loss_norm}
        return LR+LB+self.lmbda*loss_norm, losses