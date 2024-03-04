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
    def __init__(self, path2img, transform=False,aug_sw= True, aug_flip = True, sharp =False, patch_size=384):

        

        self.patch_size = patch_size
        
        self.sharp = sharp
        self.aug_sw = aug_sw
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
    
    def usm_sharp(self,img, weight=0.5, radius=50, threshold=10):
        """USM sharpening.

        Input image: I; Blurry image: B.
        1. sharp = I + weight * (I - B)
        2. Mask = 1 if abs(I - B) > threshold, else: 0
        3. Blur mask:
        4. Out = Mask * sharp + (1 - Mask) * I


        Args:
            img (Numpy array): Input image, HWC, BGR; float32, [0, 1].
            weight (float): Sharp weight. Default: 1.
            radius (float): Kernel size of Gaussian blur. Default: 50.
            threshold (int):
        """
        img = img/ 255.
        
        if radius % 2 == 0:
            radius += 1
        blur = cv2.GaussianBlur(img, (radius, radius), 0)
        residual = img - blur
        mask = np.abs(residual) * 255 > threshold
        mask = mask.astype('float32')
        soft_mask = cv2.GaussianBlur(mask, (radius, radius), 0)

        sharp = img + weight * residual
        sharp = np.clip(sharp, 0, 1)
        img = soft_mask * sharp + (1 - soft_mask) * img
        return np.uint8(img*255)
    
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

        if self.sharp:
            sharp_prop = random.randint(0, 1)
            if sharp_prop==1: HR = self.usm_sharp(HR)
            
        if self.aug_sw:
            HR = self.apply_random_mask(h, w, HR)
            
        if self.aug_flip:
            rotTimes = random.randint(0, 3)
            vFlip = random.randint(0, 1)
            hFlip = random.randint(0, 1)           

            HR = self.arguement(HR, rotTimes, vFlip, hFlip)                   
        
        LR = cv2.resize(HR,[48,48],interpolation=cv2.INTER_CUBIC)
        HR = self.transform(Image.fromarray(HR))
        LR = self.transform(Image.fromarray(LR))
        
        return LR,HR

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

        
