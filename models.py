import torch
import torch.nn.functional as F
from torch import nn, cuda
from torch.autograd import Variable
from torch.nn.utils import weight_norm
from torchsummary import summary
from utils import gen_train_mask

class SimpleNet_no_PConv(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.krnl_size = 7
        self.decoder = nn.Conv2d(1,128,self.krnl_size,bias=False)
        self.conv = nn.Conv2d(128,1,self.krnl_size,padding=(6,6),bias=False)
        #Partial Convolution Module 
    def forward(self, x, mode = 'train'):
        p = self.krnl_size-1
        x = self.decoder(x)
        latent = x
        n,c,h,w = x.shape
        x = self.conv(x)
        # print(self.wn.weight_g, self.wn.weight_g.requires_grad)
        # return x, latent
        features ={
            'latent': latent,
        }
        return x, features
    
class SimpleNet(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.krnl_size = 7
        self.decoder = nn.Conv2d(1,128,self.krnl_size,bias=False)
        self.pconv = nn.Conv2d(128,1,self.krnl_size,padding=(6,6),bias=False)
        self.wn_pconv = weight_norm(self.pconv,name='weight')
        self.wn_pconv.weight_g = nn.Parameter(torch.ones_like(self.wn_pconv.weight_g))
        self.wn_pconv.weight_g.requires_grad = False 
        self.avgpool = nn.AvgPool2d(self.krnl_size,stride=1)
        #Partial Convolution Module 
    def forward(self, x, mode = 'train'):
        p = self.krnl_size-1
        x = self.decoder(x)
        latent = x
        n,c,h,w = x.shape
        mask = torch.ones(n,1,h,w).to(x)
        x = self.wn_pconv(x)
        mask = self.avgpool(F.pad(mask,(p,p,p,p)))
        # print(self.wn.weight_g, self.wn.weight_g.requires_grad)
        # return x, latent
        features ={
            'latent': latent,
        }
        return x/mask, features

class ResBlock(nn.Module):
    def __init__(self,in_ch = 128, krnl_size = 3 ) -> None:
        super().__init__()
        self.krnl_size = krnl_size 
        self.pad_size = (self.krnl_size - 1, self.krnl_size - 1)
        self.conv = nn.Conv2d(in_ch,in_ch*2,self.krnl_size,bias=True)
        self.wn_conv = nn.utils.weight_norm(self.conv,name='weight')
        self.wn_conv.weight_g = nn.Parameter(torch.ones_like(self.wn_conv.weight_g))
        self.wn_conv.weight_g.requires_grad = False 
        self.relu = nn.ReLU()
        self.pconv = nn.Conv2d(in_ch*2,in_ch,self.krnl_size,padding =self.pad_size,bias=True)
        self.pconv.weight.data.fill_(0) #zero initialization
        self.avgpool = nn.AvgPool2d(self.krnl_size,stride=1)
    def forward(self, x):
        p = self.krnl_size-1
        z  = self.wn_conv(x)
        z = self.relu(z)
        n , _, h, w = z.shape 
        mask = torch.ones(n,1 , h, w).to(z)
        z = self.pconv(z)
        mask = self.avgpool(F.pad(mask,(p, p, p, p)))
        z = z/mask
        return z+x
    
class ResNet(nn.Module):
    def __init__(self, n_res_block) -> None:
        super().__init__()
        self.krnl_size = 7
        self.pad_size = (self.krnl_size-1, self.krnl_size-1)
        self.n_res_block = n_res_block    
        self.decoder = nn.Conv2d(1,128,self.krnl_size,bias=False)
        
        #Residual Part 
        self.res_block = nn.Sequential(*[ResBlock() for _ in range(self.n_res_block)])
        #End of Residual Part
        
        
        self.pconv = nn.Conv2d(128,1,self.krnl_size,padding=self.pad_size,bias=False)
        self.wn_pconv = nn.utils.weight_norm(self.pconv,name='weight')
        self.wn_pconv.weight_g = nn.Parameter(torch.ones_like(self.wn_pconv.weight_g))
        self.wn_pconv.weight_g.requires_grad = False 
        self.avgpool = nn.AvgPool2d(self.krnl_size,stride=1)
        
    def forward(self, x, mode = 'train'):
        x = self.decoder(x)
        p = self.krnl_size-1
        x = self.res_block(x)     
        latent = x
        n,c,h,w = x.shape
        mask = torch.ones(n,1,h,w).to(x)
        x = self.wn_pconv(x)
        mask = self.avgpool(F.pad(mask,(p,p,p,p)))
        # print(self.wn.weight_g, self.wn.weight_g.requires_grad)

        return x/mask , latent

class ResNet_PCA_Sparse_Mask(nn.Module):
    def __init__(self, n_res_block, pca_channel =1 ) -> None:
        super().__init__()
        self.pca_channel = pca_channel 
        self.krnl_size = 7
        self.pad_size = (self.krnl_size-1, self.krnl_size-1)
        self.n_res_block = n_res_block   
        self.decoder = nn.Conv2d(1,128+self.pca_channel,self.krnl_size,bias=False)
        
        #Residual Part 
        self.res_block = nn.Sequential(*[ResBlock(in_ch = 128+self.pca_channel) for _ in range(self.n_res_block)])
        #End of Residual Part
        
        
        self.pconv1 = nn.Conv2d(128,1,self.krnl_size,padding=self.pad_size,bias=False)
        self.wn_pconv1 = nn.utils.weight_norm(self.pconv1,name='weight')
        self.wn_pconv1.weight_g = nn.Parameter(torch.ones_like(self.wn_pconv1.weight_g))
        self.wn_pconv1.weight_g.requires_grad = False 
        self.wn_pconv1.weight_v = nn.Parameter(torch.randn_like(self.wn_pconv1.weight_v)) #normal distribution (m= 0,v = 1)
        
        self.pconv2 = nn.Conv2d(self.pca_channel,1,self.krnl_size,padding=self.pad_size,bias=False)
        self.wn_pconv2 = nn.utils.weight_norm(self.pconv2,name='weight')
        self.wn_pconv2.weight_g = nn.Parameter(torch.ones_like(self.wn_pconv2.weight_g))
        self.wn_pconv2.weight_g.requires_grad = False 
        self.wn_pconv2.weight_v = nn.Parameter(torch.ones_like(self.wn_pconv2.weight_v))
        
        self.avgpool = nn.AvgPool2d(self.krnl_size,stride=1)
        
    def forward(self, x, mode = 'train'):
        x = self.decoder(x)
        p = self.krnl_size-1
        x = self.res_block(x) 
        x1 = x[:,:128,:,:]
        x2 = x[:,128:,:,:]
        n,c,h,w = x.shape
        if mode == 'train':
            mask = gen_train_mask(n,h,w).to(x1)
            x1 = x1*mask 
            x2 = x2* mask
            mask = self.avgpool(F.pad(mask,(p,p,p,p)))
        else :
            mask = torch.ones(n,1,h,w).to(x1)
            mask = self.avgpool(F.pad(mask,(p,p,p,p)))

        latent = x1 
        x1 = self.wn_pconv1(x1)
        pca_latent = x2 
        x2 = self.wn_pconv2(x2)
        x = x1/mask+x2/mask
        features = {
            'out':x1/mask,
            'pca_out':x2/mask,
            'pca_latent':pca_latent,
            'latent':latent,            
            'mask' : mask,
        }
        return x, features

class ResNet_PCA(nn.Module):
    def __init__(self, n_res_block, pca_channel =1 ) -> None:
        super().__init__()
        self.pca_channel = pca_channel
        self.krnl_size = 7
        self.pad_size = (self.krnl_size-1, self.krnl_size-1)
        self.n_res_block = n_res_block   
        self.decoder = nn.Conv2d(1,128+self.pca_channel,self.krnl_size,bias=False)
        
        #Residual Part 
        self.res_block = nn.Sequential(*[ResBlock(in_ch = 128+self.pca_channel) for _ in range(self.n_res_block)])
        #End of Residual Part
        
        
        self.pconv1 = nn.Conv2d(128,1,self.krnl_size,padding=self.pad_size,bias=False)
        self.wn_pconv1 = nn.utils.weight_norm(self.pconv1,name='weight')
        self.wn_pconv1.weight_g = nn.Parameter(torch.ones_like(self.wn_pconv1.weight_g))
        self.wn_pconv1.weight_g.requires_grad = False 
        self.wn_pconv1.weight_v = nn.Parameter(torch.randn_like(self.wn_pconv1.weight_v)) #normal distribution (m= 0,v = 1)
        
        self.pconv2 = nn.Conv2d(self.pca_channel,1,self.krnl_size,padding=self.pad_size,bias=False)
        self.wn_pconv2 = nn.utils.weight_norm(self.pconv2,name='weight')
        self.wn_pconv2.weight_g = nn.Parameter(torch.ones_like(self.wn_pconv2.weight_g))
        self.wn_pconv2.weight_g.requires_grad = False 
        self.wn_pconv2.weight_v = nn.Parameter(torch.ones_like(self.wn_pconv2.weight_v))
        
        self.avgpool = nn.AvgPool2d(self.krnl_size,stride=1)
        
    def forward(self, x, mode = 'train'):
        x = self.decoder(x)
        p = self.krnl_size-1
        x = self.res_block(x) 
        x1 = x[:,:128,:,:]
        x2 = x[:,128:,:,:]
        n,c,h,w = x.shape

        mask = torch.ones(n,1,h,w).to(x1)
        mask = self.avgpool(F.pad(mask,(p,p,p,p)))

        latent = x1 
        x1 = self.wn_pconv1(x1)
        pca_latent = x2 
        x2 = self.wn_pconv2(x2)
        x = x1/mask+x2/mask
        features = {
            'out':x1/mask,
            'pca_out':x2/mask,
            'pca_latent':pca_latent,
            'latent':latent,            
            'mask' : mask,
        }
        return x, features
    
if __name__ == "__main__":
    model = ResNet_PCA(n_res_block=3)
    summary(model, (1,128,128))