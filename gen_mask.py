import torch
import matplotlib.pyplot as plt
import random 
import imageio
import io 
from tqdm import tqdm as tqdm 
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


def visualize_mask(mask, name, save = False):
    plt.figure(figsize=(5, 5))
    plt.imshow(mask.squeeze().cpu().numpy(), cmap='gray', vmin=0, vmax =1 )
    plt.axis('off')
    
    # Save the plot to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    if save == True: 
        plt.savefig(f'{name}.png', format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    
    # Clear the current figure
    plt.close()
    
    return imageio.imread(buf)

# Parameters
n = 1  # Number of images in the batch
h, w = 122, 122  # Height and width of the image
padding = 3  # Padding size

# Generate the mask
device = torch.device('cpu')
images = []
covermaps = []
covermap = torch.zeros((n,1,h,w),device = device)
for i in tqdm(range(400)):
    # Visualize the padded mask
    mask = gen_train_mask(n, h, w, device)
    # Add zero padding
    padded_mask = torch.nn.functional.pad(mask, (padding, padding, padding, padding), mode='constant', value=0)
    img = visualize_mask(mask, name= f'mask_{i}')
    images.append(img)
    covermap = covermap+ mask
    covermap_img = visualize_mask(torch.clamp(covermap,min=0,max=1), name= f'covermap_{i}')

    covermaps.append(covermap_img)
    if i == 10: 
        imageio.mimsave('train_masks.gif', images, duration=0.5) 
    if torch.all(torch.clamp(covermap,min=0,max=1)):
        imageio.mimsave('covermaps.gif', covermaps, duration=0.01)  
        break



# # Print shapes
# print(f"Original mask shape: {mask.shape}")
# print(f"Padded mask shape: {padded_mask.shape}")