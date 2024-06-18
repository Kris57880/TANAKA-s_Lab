import torch 
from tqdm import tqdm 
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from utils import tensor_to_image
# train and test code 

def train (model, dataloader, criterion, optimizer, epoch, device, exp):
    s_loss = []
    s_recon_loss = []
    s_regu_loss = []
    s_recon_basis_loss = []    
    loss = 0 
    recon_loss = 0
    regu_loss = 0
    output = None 
    latent = None 
    input = None 
    
    for i, (input, input_L) in enumerate(tqdm(dataloader)):
        input = torch.flatten(input,start_dim=0,end_dim=1)
        input_L = torch.flatten(input_L,start_dim=0,end_dim=1)
        input = input.to(device)
        input_L = input_L.to(device)
        input_H = (input-input_L).to(device)
        
        assert len(input.shape) == 4
        optimizer.zero_grad()
        output_H, latent = model(input_H)
        output = output_H+input_L
        # print(f'output_shape:{output.shape}, output_H.range = {output_H.min(), output_H.max()}')
        loss, loss_detail = criterion(output, input, latent)
        loss.backward()
        optimizer.step()
        loss = loss.item()
        recon_loss = loss_detail['Reconstruction'].item()
        regu_loss = loss_detail['Regularization'].item()
        
        s_loss.append(loss)
        s_recon_loss.append(recon_loss)
        s_regu_loss.append(regu_loss)
        
        exp.log_metric('InTrain_Loss', loss,step=i,epoch=epoch)
        exp.log_metric('InTrain_Recon_Loss', recon_loss, step=i,epoch=epoch)
        exp.log_metric('InTrain_Regu_Loss', regu_loss, step=i,epoch=epoch)
        if i %10 == 0 :
            input = tensor_to_image(input, scale = 4)
            input_L = tensor_to_image(input_L, scale = 4 ) 
            input_H = tensor_to_image(input_H, scale = 4, normalized= True)
            output = tensor_to_image(output, scale = 4)
            output_H = tensor_to_image(output_H, scale =4 ,normalized= True)
            latent = tensor_to_image(latent, scale = 4)
            
            exp.log_image(input[0].squeeze().numpy(), image_minmax=(0,1),name=f'train_{i}_target.png', step=epoch)
            exp.log_image(input_L[0].squeeze().numpy(), name=f'train_{i}_input_L.png', step=epoch)
            exp.log_image(input_H[0].squeeze().numpy(), name=f'train_{i}_input_H.png', step=epoch)
            exp.log_image(output[0].squeeze().numpy(), image_minmax=(0,1), name=f'train_{i}_output.png', step=epoch )
            exp.log_image(output_H[0].squeeze().numpy(), name=f'train_{i}_output_H.png', step=epoch )
    exp.log_metric('Train_Loss', np.mean(s_loss),step=epoch)
    exp.log_metric('Train_Recon_Loss', np.mean(s_recon_loss),step=epoch)
    exp.log_metric('Train_Regu_Loss', np.mean(s_regu_loss),step=epoch)
    

    
    return loss, loss_detail


def test(model, dataloader, criterion, device, epoch, exp, output_dir = 'result'):
    s_loss = []
    s_recon_loss = [] 
    s_regu_loss = []
    save_list = [1,20,22,26,28,54,61,72,93,95,97,99]
    for i, (input, input_L) in enumerate(tqdm(dataloader)):
        input = input.to(device)
        input_L = input_L.to(device)
        input_H = (input-input_L).to(device)
        assert len(input.shape) == 4
        output_H, latent = model(input_H)
        output = output_H+ input_L
        loss, loss_detail = criterion(input_L, input, latent)
        # loss, loss_detail = criterion(output, input, latent)
        
        input = tensor_to_image(input, scale = 4)
        input_L = tensor_to_image(input_L, scale = 4 ) 
        input_H = tensor_to_image(input_H, scale = 4, normalized= True)
        output = tensor_to_image(output, scale = 4)
        output_H = tensor_to_image(output_H, scale =4 ,normalized= True)
        latent = latent.detach().cpu().squeeze()
        
        if i in save_list:
            n_display = 16
            n_row = 4
            kernel = model.wn_pconv.weight_v.cpu().detach().numpy()[0]
            
            input_H = (input_H+input_H.min())/(input_H.max()-input_H.min())
            output_H = (output_H+output_H.min())/(output_H.max()-output_H.min())
            exp.log_image(input.squeeze().numpy(), image_minmax=(0,1),name=f'test_{i}_target.png', step=epoch)
            exp.log_image(input_L.squeeze().numpy(), name=f'test_{i}_input_L.png', step=epoch)
            exp.log_image(input_H.squeeze().numpy(),  name=f'test_{i}_input_H.png', step=epoch)
            exp.log_image(output.squeeze().numpy(), image_minmax=(0,1), name=f'test_{i}_output.png', step=epoch )
            exp.log_image(output_H.squeeze().numpy(), name=f'test_{i}_output_H.png', step=epoch )
            
            fig, axes = plt.subplots(1,2, figsize=(10,5))
            axes[0].imshow(input.squeeze().numpy(), cmap='gray')  
            axes[1].imshow(output.squeeze().numpy(), cmap='gray')  
            axes[0].set_title('Input/Target')
            axes[1].set_title('Output')
            axes[0].axis('off')
            axes[1].axis('off')
            plt.subplots_adjust(wspace=0.2, hspace=0.2)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/Test_Image_{i}.png')
            
            #plot the feature maps
            fig, axes = plt.subplots(n_row, n_display//n_row, figsize=(n_row, n_display//n_row))
            for j, ax in enumerate(axes.flat):
                ax.imshow(latent[j], cmap='gray')  # Display the image
                ax.axis('off')  # Hide the axis 
            plt.subplots_adjust(wspace=0.2, hspace=0.2)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/feature_map_{i}.png')
            exp.log_image(f'{output_dir}/feature_map_{i}.png',name =f'test_{i}_feature', step=epoch)
            
            #plot feature maps 
            
        s_loss.append(loss.item())
        s_recon_loss.append(loss_detail['Reconstruction'].item())
        s_regu_loss.append(loss_detail['Regularization'].item())


    kernel = model.wn_pconv.weight_v.cpu().detach().numpy()[0]
    # plot kernel
    fig, axes = plt.subplots(n_row, n_display//n_row, figsize=(n_row, n_display//n_row))
    # Loop through the grid and plot each image
    for i, ax in enumerate(axes.flat):
        ax.imshow(kernel[i], cmap='gray')  # Display the image
        ax.axis('off')  # Hide the axis 
    # Adjust layout with some gap
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    # Show the plot
    plt.savefig(f'{output_dir}/kernel_{epoch}.png')
    exp.log_image(f'{output_dir}/kernel_{epoch}.png',name =f'kernel.png', step=epoch)

    
    exp.log_metric('Test_Loss', np.mean(s_loss),step=epoch)
    exp.log_metric('Test_Recon_Loss', np.mean(s_recon_loss),step=epoch)
    exp.log_metric('Test_Regu_Loss', np.mean(s_regu_loss),step=epoch)
                    
    print(f"Test Epoch {epoch+1}, Loss: {np.mean(s_loss):.4f}, Recon_Loss: {np.mean(s_recon_loss):.4f}, Regu_Loss: {np.mean(s_regu_loss):.4f}")

