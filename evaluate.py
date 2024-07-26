import comet_ml 
import torch 
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from dataloader import CustomDataset 
from models import SimpleNet, ResNet, ResNet_PCA, ResNet_PCA_Sparse_Mask
from utils import load_checkpoint, find_latest_checkpoint, list_checkpoints
from train_test import train , test 
from loss import SCLoss, SCLoss_PCA, SCLoss_PCA_Sparse_Mask
import os 
from line_notify import send_message    
exp0 = {
    'model_name' : 'ResNet_PCA_Sparse_Mask',
    'side_info' : f'1_ch_lambda_{1e-3}',
    'epoch' : 125, 
    'lmbda' : 1e-3, 
    'delta' : 0.04,
    'blur_kernel' : 7, 
    'blur_sigma': 7, 
    'lr' : 1e-4, 
    'n_res_block' : 34,
    'n_pca_channel' :1, 
    # 'ExperimentName': 'DIV2K',
    'load_checkpoint' : True,
    'experiment_name' : None
}
exp1 = {
    'model_name' : 'ResNet_PCA_Sparse_Mask',
    'side_info' : f'4_ch_lambda_{1e-3}',
    'epoch' : 125, 
    'lmbda' : 1e-3, 
    'delta' : 0.04,
    'blur_kernel' : 7, 
    'blur_sigma': 7, 
    'lr' : 1e-4, 
    'n_res_block' : 34,
    'n_pca_channel' :4, 
    # 'ExperimentName': 'DIV2K',
    'load_checkpoint' : True,
    'experiment_name' : None
}

exp2 = {
    'model_name' : 'ResNet_PCA',
    'side_info' : f'1_ch_lambda_{1e-3}_new',
    'epoch' : 125, 
    'lmbda' : 1e-3, 
    'delta' : 0.04,
    'blur_kernel' : 7, 
    'blur_sigma': 7, 
    'lr' : 1e-4, 
    'n_res_block' : 34,
    'n_pca_channel' :1, 
    # 'ExperimentName': 'DIV2K',
    'load_checkpoint' : True,
    'experiment_name' : None
}
exp3 = {
    'model_name' : 'ResNet_PCA',
    'side_info' : f'4_ch_lambda_{1e-3}',
    'epoch' : 125, 
    'lmbda' : 1e-3, 
    'delta' : 0.04,
    'blur_kernel' : 7, 
    'blur_sigma': 7, 
    'lr' : 1e-4, 
    'n_res_block' : 34,
    'n_pca_channel' :4, 
    # 'ExperimentName': 'DIV2K',
    'load_checkpoint' : True,
    'experiment_name' : None
}

for hyperparams in [exp0,exp1, exp2,exp3]:
    #hyperparams 
    hyperparams['experiment_name'] = f'{hyperparams["model_name"]}_{hyperparams["side_info"]}'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Train on {device}')

    train_set = CustomDataset(root_dir=r'Dataset\DIV2K_train_HR_bw', mode = 'train', blur_kernel = hyperparams['blur_kernel'], blur_sigma = hyperparams['blur_sigma'])
    train_dataloader = DataLoader(train_set, batch_size=4, shuffle=True)
    test_set = CustomDataset(root_dir=r'Dataset\DIV2K_valid_HR_bw',  mode = 'test', blur_kernel = hyperparams['blur_kernel'], blur_sigma = hyperparams['blur_sigma'])
    test_dataloader = DataLoader(test_set, batch_size=1, shuffle=False)

    folder_name = f'checkpoints/{hyperparams["experiment_name"]}'
    test_result_path = os.path.join(folder_name, 'result')

    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
        print(folder_name +'created successfully')

    if hyperparams['model_name'] == 'ResNet' :
        model = ResNet(n_res_block=hyperparams['n_res_block']).to(device)
    elif hyperparams['model_name'] == 'SimpleNet':
        model = SimpleNet().to(device)
    elif hyperparams['model_name'] == 'ResNet_PCA':
        model = ResNet_PCA(n_res_block=hyperparams['n_res_block'], pca_channel= hyperparams['n_pca_channel']).to(device)
    elif hyperparams['model_name'] == 'ResNet_PCA_Sparse_Mask':
        model = ResNet_PCA_Sparse_Mask(n_res_block=hyperparams['n_res_block'], pca_channel= hyperparams['n_pca_channel']).to(device)

    # Define loss function and optimizer
    if hyperparams['model_name'] == 'ResNet_PCA_Sparse_Mask':
        criterion = SCLoss_PCA_Sparse_Mask(lmbda=hyperparams['lmbda'], delta = hyperparams['delta'])
    elif hyperparams['model_name']=='ResNet_PCA':
        criterion = SCLoss_PCA(lmbda=hyperparams['lmbda'], delta = hyperparams['delta'])
    else :
        criterion = SCLoss(lmbda=hyperparams['lmbda'],delta=hyperparams['delta'])
    optimizer = optim.Adam(model.parameters(), lr = hyperparams['lr'])

    # init comet 
    comet_ml.init()
    exp = comet_ml.Experiment(api_key='NGgjeGxw7n1xPCoFUz4XUU4Zv',project_name = 'partialconv')
    exp.log_parameters(hyperparams)
    exp.set_model_graph(model)
    exp.log_code('main.py')
    exp.log_code('models.py')
    exp.log_code('train_test.py')
    exp.log_code('loss.py')
    exp.log_code('dataloader.py')
    exp.set_name(f'{hyperparams["experiment_name"]}_evaluation')

    test_result_path = os.path.join(folder_name, 'test_result')
    
    test_all_ckpt = False 
    last_epoch = 0
    #train and testing 
    if hyperparams['load_checkpoint'] == True:
        checkpoint_dir = f'{folder_name}'
        
        
        if test_all_ckpt :
            # test all checkpoints 
            checkpoints  = list_checkpoints(checkpoint_dir)
            for checkpoint_path in checkpoints:
                if checkpoint_path is not None:
                    if 'PCA' in hyperparams['model_name'] :
                        model, optimizer, last_epoch, loss, recon_losses, regu_losses, pca_recon_losses = load_checkpoint(model,hyperparams['model_name'], optimizer, checkpoint_path)
                    else :
                        model, optimizer, last_epoch, loss, recon_losses, regu_losses = load_checkpoint(model,hyperparams['model_name'], optimizer, checkpoint_path)
                else :
                    print('no checkpoint found')
                    break
                test(model,test_dataloader,criterion,device,last_epoch,exp, test_result_path)

        else :
            checkpoint_path = find_latest_checkpoint(checkpoint_dir)
            if checkpoint_path is not None:
                if 'PCA' in hyperparams['model_name'] :
                    model, optimizer, last_epoch, loss, recon_losses, regu_losses, pca_recon_losses = load_checkpoint(model,hyperparams['model_name'], optimizer, checkpoint_path)
                else :
                    model, optimizer, last_epoch, loss, recon_losses, regu_losses = load_checkpoint(model,hyperparams['model_name'], optimizer, checkpoint_path)
            else :
                print('no checkpoint found')
                break
            test(model,test_dataloader,criterion,device,last_epoch,exp, test_result_path)
   
            
    exp.end()
