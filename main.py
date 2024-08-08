import comet_ml 
import torch 
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from dataloader import CustomDataset 
from models import *
from utils import load_checkpoint, find_latest_checkpoint
from train_test_simple import train , test  #for simple network 
# from train_test import train , test 
from loss import SCLoss, SCLoss_PCA, SCLoss_PCA_Sparse_Mask
import os 
from line_notify import send_message    

exp0 = {
    'model_name' : 'ZCZC',
    'side_info' : f'',
    'epoch' : 100, 
    'lmbda' : 0, 
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
exp1 = {
    'model_name' : 'RCRC',
    'side_info' : f'',
    'epoch' : 100, 
    'lmbda' : 0, 
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
    'model_name' : 'CZC',
    'side_info' : f'',
    'epoch' : 100, 
    'lmbda' : 0, 
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
exp3 = {
    'model_name' : 'CRC',
    'side_info' : f'',
    'epoch' : 100, 
    'lmbda' : 0, 
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
exp4 = {
    'model_name' : 'ZPZP',
    'side_info' : f'',
    'epoch' : 100, 
    'lmbda' : 0, 
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
exp5 = {
    'model_name' : 'CZP',
    'side_info' : f'',
    'epoch' : 100, 
    'lmbda' : 0, 
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
exp6 = {
    'model_name' : 'ZCZP',
    'side_info' : f'',
    'epoch' : 100, 
    'lmbda' : 0, 
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
for hyperparams in [exp0,exp1,exp2,exp3,exp4,exp5,exp6]:
    #hyperparams 
    # hyperparams['experiment_name'] = f'{hyperparams["model_name"]}_{hyperparams["side_info"]}'
    hyperparams['experiment_name'] = f'{hyperparams["model_name"]}_32ch'
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
    elif hyperparams['model_name'] == 'SimpleNet_no_PConv':
        model = SimpleNet_no_PConv().to(device)
    elif hyperparams['model_name'] == 'ResNet_PCA':
        model = ResNet_PCA(n_res_block=hyperparams['n_res_block'], pca_channel= hyperparams['n_pca_channel']).to(device)
    elif hyperparams['model_name'] == 'ResNet_PCA_Sparse_Mask':
        model = ResNet_PCA_Sparse_Mask(n_res_block=hyperparams['n_res_block'], pca_channel= hyperparams['n_pca_channel']).to(device)
    elif hyperparams['model_name'] == 'CZP':
        model = CZP().to(device)
    elif hyperparams['model_name'] == 'ZCZC':
        model = ZCZC().to(device)
    elif hyperparams['model_name'] == 'RCRC':
        model = RCRC().to(device)
    elif hyperparams['model_name'] == 'CZC':
        model = CZC().to(device)
    elif hyperparams['model_name'] == 'CRC':
        model = CRC().to(device)
    elif hyperparams['model_name'] == 'ZPZP':
        model = ZPZP().to(device)
    elif hyperparams['model_name'] == 'ZCZP':
        model = ZCZP().to(device)
    else :
        print(f"Model {hyperparams['model_name']} not found")
        break
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
    exp.set_name(f'{hyperparams["experiment_name"]}')

    test_result_path = os.path.join(folder_name, 'test_result')
    train_result_path = os.path.join(folder_name, 'train_result')
    
    last_epoch = 0
    #train and testing 
    if hyperparams['load_checkpoint'] == True:
        checkpoint_dir = f'{folder_name}'
        latest_checkpoint_path = find_latest_checkpoint(checkpoint_dir)
        if latest_checkpoint_path is not None:
            if 'PCA' in hyperparams['model_name'] :
                model, optimizer, last_epoch, loss, recon_losses, regu_losses, pca_recon_losses = load_checkpoint(model,hyperparams['model_name'], optimizer, latest_checkpoint_path)
            else :
                model, optimizer, last_epoch, loss, recon_losses, regu_losses = load_checkpoint(model,hyperparams['model_name'], optimizer, latest_checkpoint_path)
        else :
            print('no checkpoint found')
            if not os.path.exists(test_result_path):
                os.mkdir(test_result_path)
            if not os.path.exists(train_result_path):
                os.mkdir(train_result_path)            
                
   
    for epoch in range(last_epoch, hyperparams['epoch']+1):
        try:
            if epoch %5 == 0 :
                test(model,test_dataloader,criterion,device,epoch,exp, test_result_path)
                
            train_loss, train_loss_detail = train(model,train_dataloader,criterion,optimizer,epoch,device,exp, train_result_path)
            if epoch % 10 == 0 :#and epoch!=0:
                if 'PCA' in hyperparams['model_name']:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': train_loss,
                        'recon_losses': train_loss_detail['Reconstruction'],
                        'regu_losses': train_loss_detail['Regularization'],    
                        'pca_recon_losses': train_loss_detail['Basis_Recon'],        
                        }, f'{folder_name}/{epoch}.pth')
                else :
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': train_loss,
                        'recon_losses': train_loss_detail['Reconstruction'],
                        'regu_losses': train_loss_detail['Regularization'],    
                        }, f'{folder_name}/{epoch}.pth')
        except KeyboardInterrupt:
            exp.end()
            break 
        # except Exception as e:
        #     send_message("Error happened ! Experiment Interrupted")
        #     exp.end()
            
    exp.end()
