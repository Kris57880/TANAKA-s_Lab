import comet_ml 
import torch 
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from dataloader import CustomDataset 
from models import SimpleNet, ResNet, ResNet_PCA
from utils import load_checkpoint, find_latest_checkpoint
from train_test import train , test 
from loss import SCLoss
import os 


#hyperparams 
hyperparams = {
    'model_name' : 'ResNet_PCA',
    'epoch' : 75, 
    'lmbda' : 5e-4, 
    'delta' : 0.04,
    'blur_kernel' : 7, 
    'blur_sigma': 7, 
    'lr' : 1e-4, 
    'n_res_block' : 34,
    # 'ExperimentName': 'DIV2K',
    'load_checkpoint' : True,
    'experiment_name' : None
}
hyperparams['experiment_name'] = f'{hyperparams["model_name"]}_lambda_{hyperparams["lmbda"]}_delta_{hyperparams["delta"]}_gaussian_{hyperparams["blur_kernel"]}_{hyperparams["blur_sigma"]}'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Train on {device}')

train_set = CustomDataset(root_dir=r'Dataset\DIV2K_train_HR_bw', mode = 'train', blur_kernel = hyperparams['blur_kernel'], blur_sigma = hyperparams['blur_sigma'])
train_dataloader = DataLoader(train_set, batch_size=4, shuffle=True)
test_set = CustomDataset(root_dir=r'Dataset\DIV2K_valid_HR_bw',  mode = 'test', blur_kernel = hyperparams['blur_kernel'], blur_sigma = hyperparams['blur_sigma'])
test_dataloader = DataLoader(test_set, batch_size=1, shuffle=False)

folder_name = f'checkpoints/{hyperparams["experiment_name"]}'
if not os.path.exists(folder_name):
    os.mkdir(folder_name)

if hyperparams['model_name'] == 'ResNet' :
    model = ResNet(n_res_block=hyperparams['n_res_block']).to(device)
elif hyperparams['model_name'] == 'SimpleNet':
    model = SimpleNet().to(device)
elif hyperparams['model_name'] == 'ResNet_PCA':
    model = ResNet_PCA(n_res_block=hyperparams['n_res_block']).to(device)
    
# Define loss function and optimizer
criterion = SCLoss(lmbda=hyperparams['lmbda'], delta = hyperparams['delta'])
optimizer = optim.Adam(model.parameters(), lr = hyperparams['lr'])

# init comet 
comet_ml.init()
exp = comet_ml.Experiment(api_key='NGgjeGxw7n1xPCoFUz4XUU4Zv',project_name = 'partialconv')
exp.log_parameters(hyperparams)
exp.set_model_graph(model)
exp.log_code('main.py')
exp.log_code('train_test.py')
exp.log_code('loss.py')
exp.log_code('dataloader.py')
exp.set_name(hyperparams['experiment_name'])


last_epoch = 0
#train and testing 
if hyperparams['load_checkpoint'] == True:
    checkpoint_dir = f'{folder_name}'
    latest_checkpoint_path = find_latest_checkpoint(checkpoint_dir)
    if latest_checkpoint_path is not None:
        model, optimizer, last_epoch, loss, recon_losses, regu_losses = load_checkpoint(model, optimizer, latest_checkpoint_path)
    else :
        print('no checkpoint found')

# test_result_path = os.path.join(folder_name, 'result')
# test(model,test_dataloader,criterion,device,last_epoch,exp, test_result_path)

for epoch in range(last_epoch, hyperparams['epoch']):
    try:
        test_result_path = os.path.join(folder_name, 'result')
        if not os.path.exists(test_result_path):
            os.mkdir(test_result_path)
        train_loss, train_loss_detail = train(model,train_dataloader,criterion,optimizer,epoch,device,exp)
        if epoch %5 == 0 :
            test(model,test_dataloader,criterion,device,epoch,exp, test_result_path)
            if epoch % 10 == 0:
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
exp.end()