# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 11:58:52 2024

@author: user
"""

import cv2
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

#%%
path = 'RGB/DIV2K/*.png'
img_list = sorted(glob.glob(path))
random.shuffle(img_list)


save_path_gt = 'dataset_CIDIS_sisr_x8/thermal/train/GT'
i=0

for img_path in tqdm(img_list[:600], desc = 'train_data_gen.'):
    
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    img_num_str = str(i)
    
    if i<10 : img_num_str = '000'+img_num_str
    elif i<100 : img_num_str = '00'+img_num_str
    elif i<1000 : img_num_str = '0'+img_num_str
    
    
    
    cv2.imwrite(save_path_gt+'/DIV2K'+img_num_str+'.bmp', img)
    i+=1
#%%
path = 'RGB/Urban100/Urban100/image_SRF_4/*HR.png'
img_list = sorted(glob.glob(path))
random.shuffle(img_list)


save_path_gt = 'dataset_CIDIS_sisr_x8/thermal/train/GT'

i=0

for img_path in tqdm(img_list,desc = 'train_data_gen.'):
    
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img_num_str = str(i)
    
    if i<10 : img_num_str = '00'+img_num_str
    elif i<100 : img_num_str = '0'+img_num_str
    
    
    
    
    cv2.imwrite(save_path_gt+'/Urban'+img_num_str+'.bmp', img)
    i+=1
