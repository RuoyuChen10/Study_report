# -*- coding: utf-8 -*-  

"""
Created on 2022/05/11
@author: Ruoyu Chen
"""

import os
import random
import numpy as np

import torchvision.transforms as transforms

from PIL import Image
from torch.utils import data

class Dataset(data.Dataset):
    """
    Read datasets
    Args:
        dataset_root: the images dir path
        dataset_list: the labels
    """
    def __init__(self, dataset_list='./train.txt'):
    
        with open(dataset_list,"r") as file:
            datas = file.readlines()

        self.data = [data_.rstrip("\n") for data_ in datas]

        self.transforms = transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            # transforms.CenterCrop(196),
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self,index):
        # Sample
        sample = self.data[index]
        
        # data and label information
        splits = sample.split(' ')
        image_path = splits[0]

        data = Image.open(image_path)

        data = self.transforms(data)
        
        label = [int(x) for x in splits[1:]]

        label = np.array(label).astype(np.float32)

        return data.float(), label