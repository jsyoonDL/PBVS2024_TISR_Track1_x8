# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 18:00:30 2024

@author: yjs
"""

from PIL import Image
from torch.utils.data import Dataset
import random
import numpy as np
import glob
import cv2

import torchvision.transforms as transforms

#%% Train
class DatasetTrain(Dataset):
    def __init__(self, path2img, transform=False,aug_sw= True, aug_flip = True, patch_size=384):

        
        self.patch_size = patch_size        

        self.aug_sw = aug_sw
        self.aug_flip = aug_flip
        
        self.img_path_list = []
        for ext in ['.bmp']: #
            self.img_path_list+=sorted(glob.glob(path2img+'/*'+ext))
        
        if transform :
            self.transform = transform
        else:        
            self.transform = transforms.Compose([
                                            transforms.ToTensor(),
                                            ])            
    
        
    def apply_random_mask(self,h,w, HR):
        """Randomly masks image"""

        y1 = np.random.randint(0, h - self.patch_size, 1)[0]
        x1 = np.random.randint(0, w - self.patch_size, 1)[0]
        
        y2, x2 = y1 + self.patch_size, x1 + self.patch_size
        
        masked_HR = HR[y1:y2, x1:x2,:].copy()  
        
        return masked_HR
    
    def arguement(self, img, rotTimes, vFlip, hFlip):
        # Random rotation
        for j in range(rotTimes):
            img = np.rot90(img.copy(), axes=(0, 1))
        # Random vertical Flip
        for j in range(vFlip):
            img = img[::-1, :, :].copy()
        # Random horizontal Flip
        for j in range(hFlip):
            img = img[:, ::-1, :].copy()
        return img
    
    
    def __getitem__(self, index):

        img_path = self.img_path_list[index]       
        
        HR = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        h, w, c = HR.shape

           
        if self.aug_sw:
            HR = self.apply_random_mask(h, w, HR)
            
        if self.aug_flip:
            rotTimes = random.randint(0, 3)
            vFlip = random.randint(0, 1)
            hFlip = random.randint(0, 1)           

            HR = self.arguement(HR, rotTimes, vFlip, hFlip)                   

        HR = self.transform(Image.fromarray(HR))
         
        return HR

    def __len__(self):
        return len(self.img_path_list)


#%% val
class Dataset(Dataset):
    def __init__(self, path2img, transform=False, aug_flip = True):

        self.aug_flip = aug_flip
        
        self.img_path_list = []
        for ext in ['.bmp','.png','.jpg','tiff','tif','jpeg']: #
            self.img_path_list+=sorted(glob.glob(path2img+'/*'+ext))
        
        if transform :
            self.transform = transform
        else:        
            self.transform = transforms.Compose([
                                            transforms.ToTensor(),
                                            ])            
    
    def arguement(self, img, vFlip, hFlip):


        # Random vertical Flip
        for j in range(vFlip):
            img = img[::-1, :, :].copy()
            
        # Random horizontal Flip
        for j in range(hFlip):
            img = img[:, ::-1, :].copy()
        return img
    
    def __getitem__(self, index):

        img_path = self.img_path_list[index]
        LR = cv2.imread(img_path, cv2.IMREAD_UNCHANGED) # cv2.IMREAD_GRAYSCALE, cv2.IMREAD_UNCHANGED
        # LR = LR.astype(np.float32) / 255.
        
        HR = cv2.imread(img_path.replace('LR_x8','GT'), cv2.IMREAD_UNCHANGED)
        
        # HR = HR.astype(np.float32) / 255.
        
        
        if self.aug_flip:
            vFlip = random.randint(0, 1)
            hFlip = random.randint(0, 1)
            
            LR = self.arguement(LR, vFlip, hFlip)
            HR = self.arguement(HR, vFlip, hFlip)
       

            
        LR = self.transform(LR)
        HR = self.transform(HR)
        
        
        return LR,HR

    def __len__(self):
        return len(self.img_path_list)

        
