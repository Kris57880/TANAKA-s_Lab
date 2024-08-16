### File Organization

```
Root
|--Dataset
|   |--Train
|		|--Valid
|--Experiment  
|		|--Experiment Name
|		|		|--*.pth  #checkpoints 
|		|		|--result #train/test result image
|--model.py 
|--utils.py #some useful function to run the experiment  
|--main.py #script for setting up the experiment and running the train/test function
|--train_test.py #code about train and test function
|--train_test_simple.py #train and test for Simple model (CZC/CRC/CZP/ZCZC/RCRC/ZCZP)
|--evaluate.py #only testing (run the newest checkpoint or all checkpoints under the folder you assigned)
|--requirements.txt #install necessary libraries 
```

### How To Use

<aside>
⚠️ Since the experiment process highly relies on the functions from  [Comet ML](https://www.comet.com/site/) to record and experiment data (images, training curves, losses … etc), I strongly suggest registering an account for Comet and understanding how to use it 
If not, please write your own record function and replace any code starting with `exp`

</aside>

- Training
    
    <aside>
    ⚠️ if you want to train simple models (CZC/CRC/CZP/ZCZC/RCRC/ZCZP), please replace the import code from train_test.py to train_test_simple.py
    
    </aside>
    
    1. prepare the training config at the beginning of `main.py`
        
        *note that if any config is mark as optional (depend on the model and loss function you use), you still need to fill some random value, it will not affect the result* 
        
        ```
        exp = {
            'model_name' : 'ZCZC', #the name of model, will discuss in below section
            'side_info' : f'', #other information you want to write down for this information 
            'epoch' : 100, 
            'lmbda' : 0, #lambda value for loss function (Reconstruction Loss + lambda * Regularization loss)
            'delta' : 0.04, # (optional) delta value for Huber loss, please refer to https://pytorch.org/docs/stable/generated/torch.nn.HuberLoss.html
            'blur_kernel' : 7, #(optional)the kernel size for Frequency separation version of model 
            'blur_sigma': 7, #(optional)the gaussian blur sigma value 
            'lr' : 1e-4, #learning rate 
            'n_res_block' : 34, #number of residual blocks in model 
            'n_pca_channel' :4, #number of channel for PCA branch
            'load_checkpoint' : True, #if you want to load previous checkpoint (resume training)
            'experiment_name' : None
        }
        ```
        
    2. Directly run the `main.py`
        
        It will test every 5 epochs and save every 10 epochs 
        
    
- Testing
    - If you only want to test your model, please use `evaluate.py`
    - The procedure is the same as mentioned in the training section, instead of running `main.py`, run `evaluate.py`
    - there is an option `test_all_ckpt`  to run all the checkpoints under every checkpoint in the same experiment folder, after that, you can get the series of images showing the changes between training process.
- Visualize

### Models

There are many models in this repo, shown as below 

- model for showing the effectiveness of partial convolution in border effect, demonstrated below, the left part is their name
    
    ![Untitled](Instructions%20about%20this%20Repo%207eabe0cd6dea42568c48e11f664e8deb/Untitled.png)
    
    - this refers to the ‘simple model’ as I mentioned above
- model for leverage between image reconstruction and regularization
    
    *please refer to PartialConvolution1.drawio for detailed architecture*
    
    - SimpleNet (same as CzP model) / With CSC objective
        
        ![Untitled](Instructions%20about%20this%20Repo%207eabe0cd6dea42568c48e11f664e8deb/Untitled%201.png)
        
    - ResNet (add w/ residual block)
        
        ![Untitled](Instructions%20about%20this%20Repo%207eabe0cd6dea42568c48e11f664e8deb/96018e29-651c-4905-ac06-6e912edc08d6.png)
        
    - ResNet/ Separate channel by frequency
        
        ![Untitled](Instructions%20about%20this%20Repo%207eabe0cd6dea42568c48e11f664e8deb/29404843-2b87-4718-bd44-ba56dd0678e3.png)
        
    - ResNet/ Separate channel by PCA
        
        ![Untitled](Instructions%20about%20this%20Repo%207eabe0cd6dea42568c48e11f664e8deb/Untitled%202.png)
        
    - ResNet/ Separate channel by PCA/ add Sparse Mask
        
        ![Untitled](Instructions%20about%20this%20Repo%207eabe0cd6dea42568c48e11f664e8deb/Untitled%203.png)
