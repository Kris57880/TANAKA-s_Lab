import torch 
from tqdm import tqdm 
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
import numpy as np
from utils import normalize

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
        output,features = model(input,mode='train')
        loss, loss_detail = criterion(output, input, features, mode= 'train')
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
            save_image(input[0].detach().cpu().squeeze(), f'{output_dir}/input_{i}.png')
            save_image(output[0].detach().cpu().squeeze(), f'{output_dir}/output_{i}.png')
            save_image(features['pca_out'][0].detach().cpu().squeeze(), f'{output_dir}/pca_out_{i}.png')
            save_image(features['out'][0].detach().cpu().squeeze()+0.5, f'{output_dir}/out_{i}.png')
            save_image(normalize(features['latent'][0].detach().cpu().squeeze()[:4].unsqueeze(1)), f'{output_dir}/feature_map_{i}.png',nrow = 2) 

            if features['pca_latent'][0].detach().cpu().squeeze().shape[0] ==1 :
                save_image(normalize(features['pca_latent'][0].detach().cpu().squeeze()), f'{output_dir}/pca_feature_map_{i}.png')
            else:
                save_image(normalize(features['pca_latent'][0].detach().cpu().squeeze()[:4].unsqueeze(1)), f'{output_dir}/pca_feature_map_{i}.png', nrow=2)

            save_image(features['mask'][0].detach().cpu().squeeze(),f'{output_dir}/mask_{i}.png')
            exp.log_image(f'{output_dir}/input_{i}.png',name=f'train_{i}_target.png', step=epoch)
            exp.log_image(f'{output_dir}/output_{i}.png', name=f'train_{i}_output.png', step=epoch )
            exp.log_image(f'{output_dir}/pca_out_{i}.png', name=f'train_{i}_pca_out.png', step=epoch )
            exp.log_image(f'{output_dir}/out_{i}.png', name=f'train_{i}_out.png', step=epoch )
            exp.log_image(f'{output_dir}/feature_map_{i}.png', name=f'train_{i}_feature.png', step=epoch)
            exp.log_image(f'{output_dir}/pca_feature_map_{i}.png', name=f'train_{i}_pca_feature.png', step=epoch)
            exp.log_image(f'{output_dir}/mask_{i}.png', name=f'train_{i}_mask.png', step=epoch)
            
            #plot csc kernel 
            kernel = normalize(model.wn_pconv1.weight_v.detach().cpu()[0][:4].unsqueeze(1))
            save_image(kernel, f'{output_dir}/kernel_{epoch}_{i}.png',nrow = 2)
            exp.log_image(f'{output_dir}/kernel_{epoch}_{i}.png',name =f'kernel_{i}.png', step=epoch)

            # plot pca kernel 
            if model.wn_pconv2.weight_v.detach().cpu()[0].shape[0] == 1 : 
                # display 1 channel 
                kernel = normalize(model.wn_pconv2.weight_v.detach().cpu()[0].unsqueeze(1))
                save_image(kernel, f'{output_dir}/pca_kernel_{epoch}_{i}.png')
            else : 
                # display 4 channel 
                kernel =  normalize(model.wn_pconv2.weight_v.detach().cpu()[0][:4].unsqueeze(1))
                save_image(kernel, f'{output_dir}/pca_kernel_{epoch}_{i}.png',nrow = 2)
            
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
        output, features = model(input,mode='test')
        loss, loss_detail = criterion(output, input, features, mode = 'test')
        # test code 
        # original_norm = torch.norm(input)
        # feature_norm = torch.norm(features['pca_latent'])+torch.norm(features['latent'])
        # print(f'image {i}: original norm: {original_norm}, feature norm: {feature_norm}')
        # end of test 
        if i in save_list:
            save_image(input.detach().cpu().squeeze(), f'{output_dir}/input_{i}.png')
            save_image(output.detach().cpu().squeeze(), f'{output_dir}/output_{i}.png')
            save_image(features['pca_out'].detach().cpu().squeeze(), f'{output_dir}/pca_out_{i}.png')
            save_image(features['out'].detach().cpu().squeeze()+0.5, f'{output_dir}/out_{i}.png')
            save_image(normalize(features['latent'].detach().cpu().squeeze()[:4].unsqueeze(1)), f'{output_dir}/feature_map_{i}.png',nrow = 2)
            if features['pca_latent'].detach().cpu().shape[1] ==1 :
                save_image(normalize(features['pca_latent'].detach().cpu().squeeze()), f'{output_dir}/pca_feature_map_{i}.png')
            else:
                save_image(normalize(features['pca_latent'].detach().cpu().squeeze()[:4].unsqueeze(1)), f'{output_dir}/pca_feature_map_{i}.png', nrow=2)
                
            # save_image(features['mask'][0].detach().cpu().squeeze(),f'{output_dir}/mask_{i}.png')
            exp.log_image(f'{output_dir}/input_{i}.png',name=f'test_{i}_target.png', step=epoch)
            exp.log_image(f'{output_dir}/output_{i}.png', name=f'test_{i}_output.png', step=epoch )
            exp.log_image(f'{output_dir}/pca_out_{i}.png', name=f'test_{i}_pca_out.png', step=epoch )
            exp.log_image(f'{output_dir}/out_{i}.png', name=f'test_{i}_out.png', step=epoch )
            exp.log_image(f'{output_dir}/feature_map_{i}.png', name=f'test_{i}_feature', step=epoch)
            exp.log_image(f'{output_dir}/pca_feature_map_{i}.png', name=f'test_{i}_pca_feature', step=epoch)
            # exp.log_image(f'{output_dir}/mask_{i}.png', name=f'test_{i}_mask.png', step=epoch)
       
        s_loss.append(loss.item())
        s_recon_loss.append(loss_detail['Reconstruction'].item())
        s_regu_loss.append(loss_detail['Regularization'].item())
        s_recon_basis_loss.append(loss_detail['Basis_Recon'].item())
        
    #plot csc kernel 
    kernel = normalize(model.wn_pconv1.weight_v.detach().cpu()[0][:4].unsqueeze(1))
    save_image(kernel, f'{output_dir}/kernel_{epoch}_{i}.png',nrow = 2)
    exp.log_image(f'{output_dir}/kernel_{epoch}_{i}.png',name =f'kernel_{i}.png', step=epoch)

    # plot pca kernel 
    if model.wn_pconv2.weight_v.detach().cpu()[0].shape[0] == 1 : 
        # display 1 channel 
        kernel = normalize(model.wn_pconv2.weight_v.detach().cpu()[0].unsqueeze(1))
        save_image(kernel, f'{output_dir}/pca_kernel_{epoch}_{i}.png')
    else : 
        # display 4 channel 
        kernel =  normalize(model.wn_pconv2.weight_v.detach().cpu()[0][:4].unsqueeze(1))
        save_image(kernel, f'{output_dir}/pca_kernel_{epoch}_{i}.png',nrow = 2)
    
    exp.log_image(f'{output_dir}/pca_kernel_{epoch}_{i}.png',name =f'pca_kernel_{i}.png', step=epoch)
    
    
    exp.log_metric('Test_Loss', np.mean(s_loss),step=epoch)
    exp.log_metric('Test_Recon_Loss', np.mean(s_recon_loss),step=epoch)
    exp.log_metric('Test_Regu_Loss', np.mean(s_regu_loss),step=epoch)
    exp.log_metric('Test_Basis_Recon_Loss', np.mean(s_recon_basis_loss),step=epoch)

    print(f"Test Epoch {epoch+1}, Loss: {np.mean(s_loss):.4f}, Recon_Loss: {np.mean(s_recon_loss):.4f}, Regu_Loss: {np.mean(s_regu_loss):.4f}")
