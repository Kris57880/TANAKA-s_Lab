import torch
from torch import nn
import torch.nn.functional as F

class SCLoss(nn.Module):
    def __init__(self, lmbda, delta):
        super(SCLoss, self).__init__()
        self.lmbda = lmbda
        self.delta = delta
    def forward(self, input, target, features, mode = 'train'):
        assert input.shape==target.shape
        LR = F.huber_loss(input,target,delta=self.delta)
        if mode == 'train':
            loss_norm = torch.norm(features['latent'])
        else : 
            loss_norm = torch.norm(features['latent'])
        losses = {'Reconstruction':LR,'Regularization':loss_norm}
        return LR+self.lmbda*loss_norm, losses
    
class SCLoss_PCA(nn.Module):
    def __init__(self, lmbda, delta):
        super(SCLoss_PCA, self).__init__()
        self.lmbda = lmbda
        self.delta = delta
    def forward(self, input, target, features, mode = 'train'):
        assert input.shape==target.shape
        loss_l2 = 0.5*torch.norm(target-input)**2
        LR = F.huber_loss(input,target,delta=self.delta)
        LB = F.huber_loss(features['pca_out'],target,delta=self.delta)
        if mode == 'train':
            loss_norm = torch.norm(features['latent'])
        else : 
            loss_norm = torch.norm(features['latent'])
        losses = {'Reconstruction':LR, 'Basis_Recon':LB, 'Regularization':loss_norm}
        return LR+LB+self.lmbda*loss_norm, losses
    
class SCLoss_PCA_Sparse_Mask(nn.Module):
    def __init__(self, lmbda, delta):
        super(SCLoss_PCA_Sparse_Mask, self).__init__()
        self.lmbda = lmbda
        self.delta = delta
    def forward(self, input, target, features, mode = 'train'):
        assert input.shape==target.shape
        loss_l2 = 0.5*torch.norm(target-input)**2
        LR = F.huber_loss(input,target,delta=self.delta)
        LB = F.huber_loss(features['pca_out'],target,delta=self.delta)
        if mode == 'train':
            loss_norm = torch.norm(features['latent'])/torch.norm(features['mask'])
        else : 
            loss_norm = torch.norm(features['latent'])
        losses = {'Reconstruction':LR, 'Basis_Recon':LB, 'Regularization':loss_norm}
        return LR+LB+self.lmbda*loss_norm, losses