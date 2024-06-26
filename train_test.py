import torch 
from tqdm import tqdm 
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
import numpy as np
from utils import tensor_to_image

# train and test code 


def train (model, dataloader, criterion, optimizer, epoch, device, exp, output_dir = 'result'):
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
    
    for i, input in enumerate(tqdm(dataloader)):
        input = torch.flatten(input,start_dim=0,end_dim=1)
        input = input.to(device)
        
        assert len(input.shape) == 4
        optimizer.zero_grad()
        output,features = model(input)
        # print(f'output_shape:{output.shape}, output_H.range = {output_H.min(), output_H.max()}')
        loss, loss_detail = criterion(output, input, features['latent'], features['pca_out'])
        loss.backward()
        optimizer.step()
        loss = loss.item()
        recon_loss = loss_detail['Reconstruction'].item()
        regu_loss = loss_detail['Regularization'].item()
        basis_loss = loss_detail['Basis_Recon'].item()
        s_loss.append(loss)
        s_recon_loss.append(recon_loss)
        s_regu_loss.append(regu_loss)
        s_recon_basis_loss.append(basis_loss)
        
        exp.log_metric('InTrain_Loss', loss,step=i,epoch=epoch)
        exp.log_metric('InTrain_Recon_Loss', recon_loss, step=i,epoch=epoch)
        exp.log_metric('InTrain_Regu_Loss', regu_loss, step=i,epoch=epoch)
        exp.log_metric('InTrain_Basis_Recon_Loss', basis_loss, step=i,epoch=epoch)
        
        if i %100 == 0 :
            # input = tensor_to_image(input, scale = 4)
            # output = tensor_to_image(output, scale = 4)
            # pca_out = tensor_to_image(features['pca_out'], scale= 4 )
            # out = tensor_to_image(features['out'], scale= 4 )
            
            # exp.log_image(input[0].squeeze().numpy(), image_minmax=(0,1),name=f'train_{i}_target.png', step=epoch)
            # exp.log_image(output[0].squeeze().numpy(), image_minmax=(0,1), name=f'train_{i}_output.png', step=epoch )
            # exp.log_image(pca_out[0].squeeze().numpy(), image_minmax=(0,1), name=f'train_{i}_pca_out.png', step=epoch )
            # exp.log_image(out[0].squeeze().numpy(), image_minmax=(0,1), name=f'train_{i}_out.png', step=epoch )
            save_image(input[0].detach().cpu().squeeze(), f'{output_dir}/input_{i}.png')
            save_image(output[0].detach().cpu().squeeze(), f'{output_dir}/output_{i}.png')
            save_image(features['pca_out'][0].detach().cpu().squeeze(), f'{output_dir}/pca_out_{i}.png')
            save_image(features['out'][0].detach().cpu().squeeze()+0.5, f'{output_dir}/out_{i}.png')
            save_image(features['latent'][0].detach().cpu().squeeze()[:4].unsqueeze(1)+0.5, f'{output_dir}/feature_map_{i}.png',nrow = 2)
            save_image(features['pca_latent'][0].detach().cpu().squeeze()+0.5, f'{output_dir}/pca_feature_map_{i}.png')
            
            # print(f'input range:{input.min(),input.max()}')
            exp.log_image(f'{output_dir}/input_{i}.png',name=f'train_{i}_target.png', step=epoch)
            exp.log_image(f'{output_dir}/output_{i}.png', name=f'train_{i}_output.png', step=epoch )
            exp.log_image(f'{output_dir}/pca_out_{i}.png', name=f'train_{i}_pca_out.png', step=epoch )
            exp.log_image(f'{output_dir}/out_{i}.png', name=f'train_{i}_out.png', step=epoch )
            exp.log_image(f'{output_dir}/feature_map_{i}.png', name=f'train_{i}_feature', step=epoch)
            exp.log_image(f'{output_dir}/pca_feature_map_{i}.png', name=f'train_{i}_pca_feature', step=epoch)
            
            kernel = model.wn_pconv1.weight_v.detach().cpu()[0][:4].unsqueeze(1)+0.5
            save_image(kernel, f'{output_dir}/kernel_{epoch}_{i}.png',nrow = 2)
            exp.log_image(f'{output_dir}/kernel_{epoch}_{i}.png',name =f'kernel_{i}.png', step=epoch)

            # plot pca kernel 
            kernel = model.wn_pconv2.weight_v.detach().cpu()[0]
            save_image(kernel, f'{output_dir}/pca_kernel_{epoch}_{i}.png')
            exp.log_image(f'{output_dir}/pca_kernel_{epoch}_{i}.png',name =f'pca_kernel_{i}.png', step=epoch)
            
            # plt.close('all')
    exp.log_metric('Train_Loss', np.mean(s_loss),step=epoch)
    exp.log_metric('Train_Recon_Loss', np.mean(s_recon_loss),step=epoch)
    exp.log_metric('Train_Regu_Loss', np.mean(s_regu_loss),step=epoch)
    exp.log_metric('Train_Basis_Recon_Loss', np.mean(s_recon_basis_loss),step=epoch)
    

    
    return loss, loss_detail


def test(model, dataloader, criterion, device, epoch, exp, output_dir = 'result'):
    s_loss = []
    s_recon_loss = [] 
    s_regu_loss = []
    s_recon_basis_loss = []
    save_list = [1,20,22,26,28,54,61,72,93,95,97,99]
    

    for i, input in enumerate(tqdm(dataloader)):
        input = input.to(device)
        assert len(input.shape) == 4
        output, features = model(input)
        loss, loss_detail = criterion(output, input, features['latent'], features['pca_out'])

        
        # input = tensor_to_image(input, scale = 4,normalized=True)
        # output = tensor_to_image(output, scale = 4,normalized=True)
        # pca_out = tensor_to_image(features['pca_out'], scale =4,normalized=True )
        # out = tensor_to_image(features['out'], scale = 4,normalized=True)
        # pca_latent = features['pca_latent'].detach().cpu().squeeze()
        # latent = features['latent'].detach().cpu().squeeze()
        if i in save_list:
            n_display = 16
            n_row = 4
            
            
            save_image(input.detach().cpu().squeeze(), f'{output_dir}/input_{i}.png')
            save_image(output.detach().cpu().squeeze(), f'{output_dir}/output_{i}.png')
            save_image(features['pca_out'].detach().cpu().squeeze(), f'{output_dir}/pca_out_{i}.png')
            save_image(features['out'].detach().cpu().squeeze()+0.5, f'{output_dir}/out_{i}.png')
            save_image(features['latent'].detach().cpu().squeeze()[:4].unsqueeze(1)+0.5, f'{output_dir}/feature_map_{i}.png',nrow = 2)
            save_image(features['pca_latent'].detach().cpu().squeeze()+0.5, f'{output_dir}/pca_feature_map_{i}.png')
            # print(f'input range:{input.min(),input.max()}')
            exp.log_image(f'{output_dir}/input_{i}.png',name=f'test_{i}_target.png', step=epoch)
            exp.log_image(f'{output_dir}/output_{i}.png', name=f'test_{i}_output.png', step=epoch )
            exp.log_image(f'{output_dir}/pca_out_{i}.png', name=f'test_{i}_pca_out.png', step=epoch )
            exp.log_image(f'{output_dir}/out_{i}.png', name=f'test_{i}_out.png', step=epoch )
            exp.log_image(f'{output_dir}/feature_map_{i}.png', name=f'test_{i}_feature', step=epoch)
            exp.log_image(f'{output_dir}/pca_feature_map_{i}.png', name=f'test_{i}_pca_feature', step=epoch)
            
            
            # fig, axes = plt.subplots(1,4, figsize=(10,5))
            # axes[0].imshow(input.squeeze().numpy(), cmap='gray')  
            # axes[1].imshow(output.squeeze().numpy(), cmap='gray')  
            # axes[2].imshow(pca_out.squeeze().numpy(), cmap='gray')  
            # axes[3].imshow(out.squeeze().numpy(), cmap='gray')  
            # axes[0].set_title('Input/Target')
            # axes[1].set_title('Output')
            # axes[2].set_title('pca_out')
            # axes[3].set_title('out')
            # axes[0].axis('off')
            # axes[1].axis('off')
            # axes[2].axis('off')
            # axes[3].axis('off')
            # plt.subplots_adjust(wspace=0.2, hspace=0.2)
            # plt.tight_layout()
            # plt.savefig(f'{output_dir}/Test_Image_{i}.png')
            # plt.clf()
            
            #plot the feature maps
            # fig, axes = plt.subplots(n_row, n_display//n_row, figsize=(n_row, n_display//n_row))
            # for j, ax in enumerate(axes.flat):
            #     ax.imshow(latent[j], cmap='gray')  # Display the image
            #     ax.axis('off')  # Hide the axis 
            # plt.subplots_adjust(wspace=0.2, hspace=0.2)
            # plt.tight_layout()
            # plt.savefig(f'{output_dir}/feature_map_{i}.png')
            # exp.log_image(f'{output_dir}/feature_map_{i}.png',name =f'test_{i}_feature', step=epoch)
            # plt.clf()
            
            
            #plot pca features
            # fig, axes = plt.subplots(2, 2, figsize=(2, 2))
            # for j, ax in enumerate(axes.flat):
            #     ax.imshow(pca_latent[j], cmap='gray')  # Display the image
            #     ax.axis('off')  # Hide the axis 
            # plt.subplots_adjust(wspace=0.2, hspace=0.2)
            # plt.tight_layout()
            # plt.savefig(f'{output_dir}/pca_feature_map_{i}.png')
            # exp.log_image(f'{output_dir}/pca_feature_map_{i}.png',name =f'test_{i}_pca_feature', step=epoch)
            # plt.clf()
            # #plot feature maps 
            
        s_loss.append(loss.item())
        s_recon_loss.append(loss_detail['Reconstruction'].item())
        s_regu_loss.append(loss_detail['Regularization'].item())
        s_recon_basis_loss.append(loss_detail['Basis_Recon'].item())

    # plt.close('all')
    kernel = model.wn_pconv1.weight_v.detach().cpu()[0][:4].unsqueeze(1)+0.5
    save_image(kernel, f'{output_dir}/kernel_{epoch}.png',nrow = 2)
    # plot kernel
    # fig, axes = plt.subplots(2,2, figsize=(2,2))
    # # Loop through the grid and plot each image
    # for i, ax in enumerate(axes.flat):
    #     ax.imshow(kernel[i], cmap='gray')  # Display the image
    #     ax.axis('off')  # Hide the axis 
    # # Adjust layout with some gap
    # plt.subplots_adjust(wspace=0.2, hspace=0.2)
    # # Show the plot
    # plt.savefig(f'{output_dir}/kernel_{epoch}.png')
    exp.log_image(f'{output_dir}/kernel_{epoch}.png',name =f'kernel.png', step=epoch)



    # plot pca kernel 
    kernel = model.wn_pconv2.weight_v.detach().cpu()[0]
    save_image(kernel, f'{output_dir}/pca_kernel_{epoch}.png')
    # fig, axes = plt.subplots(2,2, figsize=(2,2))
    # # Loop through the grid and plot each image
    # for i, ax in enumerate(axes.flat):
    #     ax.imshow(kernel[i], cmap='gray')  # Display the image
    #     ax.axis('off')  # Hide the axis 
    # # Adjust layout with some gap
    # plt.subplots_adjust(wspace=0.2, hspace=0.2)
    # # Show the plot
    # plt.savefig(f'{output_dir}/pca_kernel_{epoch}.png')
    exp.log_image(f'{output_dir}/pca_kernel_{epoch}.png',name =f'pca_kernel.png', step=epoch)
    # plt.close('all')

    
    exp.log_metric('Test_Loss', np.mean(s_loss),step=epoch)
    exp.log_metric('Test_Recon_Loss', np.mean(s_recon_loss),step=epoch)
    exp.log_metric('Test_Regu_Loss', np.mean(s_regu_loss),step=epoch)
    exp.log_metric('Test_Basis_Recon_Loss', np.mean(s_recon_basis_loss),step=epoch)

    print(f"Test Epoch {epoch+1}, Loss: {np.mean(s_loss):.4f}, Recon_Loss: {np.mean(s_recon_loss):.4f}, Regu_Loss: {np.mean(s_regu_loss):.4f}")

