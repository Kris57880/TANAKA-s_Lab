import torch 
from torch.utils.data import Dataset
from torchvision import transforms
from utils import RandomTranspose
import os 
from PIL import Image

train_crop_size = 64
test_crop_size = 256
n_crop = 8
train_data_transform = transforms.Compose([
    transforms.RandomCrop((train_crop_size, train_crop_size)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    RandomTranspose(p=0.5),
])
test_data_transform = transforms.Compose([
    transforms.CenterCrop((test_crop_size, test_crop_size)),
])


class CustomDataset(Dataset):
    def __init__(self, root_dir, mode = 'train', blur_kernel = None, blur_sigma = None):
        self.root_dir = root_dir
        self.mode = mode
        self.crop_size = train_crop_size
        self.n_crop = n_crop
        if self.mode == 'train':
            self.transform = train_data_transform
        else:
            self.transform = test_data_transform
            
        self.image_files = [f for f in os.listdir(root_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.blur_transform = transforms.GaussianBlur(blur_kernel,blur_sigma)
        self.toTensor = transforms.ToTensor()

    
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name)
        if self.mode == 'train':
            images = torch.empty((self.n_crop,1,self.crop_size,self.crop_size))
            blur_images = torch.empty((self.n_crop,1,self.crop_size,self.crop_size))
            for i in range(self.n_crop):
                if self.transform:
                    train_img = self.transform(image)
                    blur_image = self.blur_transform(train_img)
                    images[i] = self.toTensor(train_img)
                    blur_images[i] = self.toTensor(blur_image)
                    
            return images, blur_images
        else :
            test_image = self.transform(image)
            test_blur_image = self.blur_transform(test_image)
            test_image = self.toTensor(test_image)
            test_blur_image = self.toTensor(test_blur_image)
            return test_image,  test_blur_image
    
