from PIL import Image
import numpy as np
import os
from torch.utils.data import Dataset
import math
# import cv2
import torchvision
import torch


train_file = os.getcwd() +'/data/train_labels.txt'
# val_file = os.getcwd() + '/data/val_labels.txt'
val_file = os.getcwd() + '/data/test_labels.txt'


class load_data(Dataset):
    def __init__(self, transform=None, phase_train=True, data_dir=None,phase_test=False):

        self.phase_train = phase_train
        self.phase_test = phase_test
        self.transform = transform

        try:
            with open(train_file, 'r') as f:
                self.dir_train = f.read().splitlines()
            with open(val_file, 'r') as f:
                self.dir_val = f.read().splitlines()
#             with open(test_file_file, 'r') as f:
#                 self.dir_test = f.read().splitlines()
                
        except:
            print('can not open files, may be filelist is not exist')
            exit()

    def __len__(self):
        if self.phase_train:
            return len(self.dir_train)
        else:
            return len(self.dir_val)

    def __getitem__(self, idx):
        if self.phase_train:
#             print(self.dir_train[idx])
#             print(type(self.dir_train))
            line  = self.dir_train[idx].split()
            image_dir = line[0]
            label = int(line[1])
        else:
            line  = self.dir_val[idx].split()
            image_dir = line[0]
            label = int(line[1])

        image = Image.open(image_dir)
        image = image.convert('RGB')

        if self.transform:
            image = self.transform(image)
        if self.phase_train:
            return image,label
        else:
            return image,label
        
class ColorAugmentation(object):
    def __init__(self):
        self.eig_vec = torch.Tensor([
            [0.4009, 0.7192, -0.5675],
            [-0.8140, -0.0045, -0.5808],
            [0.4203, -0.6948, -0.5836],
        ])
        self.eig_val = torch.Tensor([[0.2175, 0.0188, 0.0045]])

    def __call__(self, tensor):
        assert tensor.size(0) == 3
        alpha = torch.normal(mean=torch.zeros_like(self.eig_val)) * 0.1
        quatity = torch.mm(self.eig_val * alpha, self.eig_vec)
        tensor = tensor + quatity.view(3, 1, 1)
        return tensor
