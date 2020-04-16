import os
import PIL
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms

class CancerDataset(Dataset):
    '''Dataset of IDC Breast Cancer images'''

    def __init__(self,root=".",annotation_file="annotations.txt",transform=None):
        self.root = root
        self.annotation_file = annotation_file
        self.transform = transform

        with open(self.annotation_file,'r') as f:
            img_files = f.readlines()
            
        self.img_files = [img_file[:-1] for img_file in img_files]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self,index):
        image = Image.open(self.img_files[index])
        label = int(self.img_files[index].split(".")[0][-1])

        if self.transform:
            image = self.transform(image)

        return image,label


