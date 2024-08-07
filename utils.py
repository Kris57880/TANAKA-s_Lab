import os 
import torch 
import comet_ml
from PIL import Image, ImageSequence
import torch.nn.functional as F
from torchvision.utils import save_image 
import random

class RandomTranspose(torch.nn.Module):
    """Transpose the given image randomly with a given probability.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions

    Args:
        p (float): probability of the image being transposed. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be transposed.

        Returns:
            PIL Image or Tensor: Randomly transposed image.
        """
        if torch.rand(1) < self.p:
            return img.transpose(Image.TRANSPOSE)
        return img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"

def gen_train_mask(n, h, w, device='cpu'):
    # Initialize mask with zeros
    mask = torch.zeros(n, 1, h, w, device=device)
    
    # Fill all four corners with 1
    mask[:, :, 0, 0] = 1
    mask[:, :, 0, -1] = 1
    mask[:, :, -1, 0] = 1
    mask[:, :, -1, -1] = 1
    
    stride = 7
    offset_x = random.randint(1,7)
    offset_y = random.randint(1,7)
    current_x = offset_x
    current_y = offset_y
    
    while current_x < w:
        mask[:,:,0, current_x] = 1 
        while current_y < h:
            mask[:,:, current_y, current_x] =1 
            current_y += stride
    
            
        mask[:,:,-1,current_x] = 1 
        current_y = offset_y
        current_x+= stride
        
    while current_y <h : 
        mask[:,:, current_y, 0] =1
        mask[:,:, current_y, -1] =1
        current_y += stride
        
    return mask.float()

def tensor_to_image(x, scale = 1, normalized = False):
    x = x.detach().cpu()
    if scale!= 1 :
        x = F.interpolate(x,scale_factor=scale, mode = 'bicubic')
    if normalized:
        x = (x-x.min())/(x.max()-x.min())
    return x 
    
def find_latest_checkpoint(checkpoint_dir):
    # List all files in the checkpoint directory
    all_files = os.listdir(checkpoint_dir)
    
    # Filter out files that do not match the checkpoint naming convention
    checkpoint_files = [f for f in all_files if f.endswith('.pth')]
    
    if not checkpoint_files:
        return None
    
    # Sort the checkpoint files based on modification time
    checkpoint_files.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)), reverse=True)
    
    # The latest checkpoint file is the first one in the sorted list
    latest_checkpoint = checkpoint_files[0]
    return os.path.join(checkpoint_dir, latest_checkpoint)

import os

def list_checkpoints(checkpoint_dir):
    # List all files in the checkpoint directory
    all_files = os.listdir(checkpoint_dir)
    
    # Filter out files that do not match the checkpoint naming convention
    checkpoint_files = [f for f in all_files if f.endswith('.pth')]
    
    if not checkpoint_files:
        return []
    
    # Sort the checkpoint files based on file size
    checkpoint_files.sort(key=lambda x: os.path.getsize(os.path.join(checkpoint_dir, x)))
    
    # Get full paths for the sorted checkpoint files
    sorted_checkpoints = [os.path.join(checkpoint_dir, f) for f in checkpoint_files]
    
    return sorted_checkpoints

def load_checkpoint(model, model_name, optimizer, checkpoint_path):
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path)

    # Load model state dictionary
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state dictionary
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load other training parameters
    epoch = checkpoint['epoch']
    print(f'load from checkpoint: {epoch}') 

    loss = checkpoint['loss']
    recon_losses = checkpoint['recon_losses']
    regu_losses = checkpoint['regu_losses']
    if 'PCA' in model_name:
        pca_recon_losses = checkpoint['pca_recon_losses']
        return model, optimizer, epoch, loss, recon_losses, regu_losses, pca_recon_losses
        
    return model, optimizer, epoch, loss, recon_losses, regu_losses

def normalize(input):
    max  = (torch.abs(input)).max()
    return input/max * 0.5 + 0.5 