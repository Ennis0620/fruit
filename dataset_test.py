import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import numpy as np
import cv2
import PIL.Image as Image
import re
from natsort import natsorted 

class test_dataset(Dataset):
    def __init__(self, img_dir, transform=None):
        super(test_dataset,self).__init__()
        self.CURREN_PATH =  os.path.dirname(__file__)

        self.img_dir = f'{img_dir}'
        self.img_len = os.listdir(self.img_dir)
        self.transform = transform
        self.img_path_list = []
        self.img_name_list= []
        img_dir_imgs = os.listdir(self.img_dir)
        img_dir_imgs = natsorted(img_dir_imgs)
        for img_name in img_dir_imgs:
            full_name = f'{self.img_dir}/{img_name}'
            self.img_path_list.append(full_name)
            self.img_name_list.append(img_name)

    def __len__(self):
        return len(self.img_len)

    def __getitem__(self, idx):
        if self.transform:
            #img = cv2.imread(self.img_path_list[idx])
            #img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            #img = read_image(self.img_path_list[idx])
            img = Image.open((self.img_path_list[idx]))
            #print(img.shape)
            image = self.transform(img)
        
            return image, self.img_name_list[idx]
	
    
