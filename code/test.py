# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 10:00:47 2024

@author: user
"""

import torch
import cv2
import numpy as np
import os
import random
import glob
from tqdm import tqdm

import torchvision.transforms as transforms


from torchvision.transforms.functional import to_pil_image

from torch.autograd import Variable
from SwinIR import SwinIR
#%%
def set_seed(seed = 0):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    # Set a fixed value for the hash seedTrue
    os.environ['PYTHONHASHSEED'] = str(seed)
#%%
def GEOMETERY_ENSEMBLE(upsampler, img, outscale):
    flip = [False, True]
    rotate = [0,90,180,270]

    height,width,channel = img.shape
    ensemble = np.zeros((height*outscale,width*outscale,channel))

    for f in flip:
        for r in rotate:
            tmp = img
            if f:
                tmp = cv2.flip(tmp,1)
            if r == 90:
                tmp = cv2.rotate(tmp, cv2.ROTATE_90_CLOCKWISE)
            elif r == 180:
                tmp = cv2.rotate(tmp, cv2.ROTATE_180)
            elif r == 270:
                tmp = cv2.rotate(tmp, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
            tmp = tmp.astype(np.float32) / 255.
            tmp = transforms.ToTensor()(tmp)
            tmp = tmp.unsqueeze(0)
            tmp = Variable(tmp.to(device))
            output = upsampler(tmp)
            output = torch.clamp(output,0,1)
            output = to_pil_image(output[0])
            output = np.array(output)
            
            if f:
                output = cv2.flip(output,1)
                if r == 90:
                    output = cv2.rotate(output, cv2.ROTATE_90_CLOCKWISE)
                elif r == 180:
                    output = cv2.rotate(output, cv2.ROTATE_180)
                elif r == 270:
                    output = cv2.rotate(output, cv2.ROTATE_90_COUNTERCLOCKWISE)
            else:
                if r == 90:
                    output = cv2.rotate(output, cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif r == 180:
                    output = cv2.rotate(output, cv2.ROTATE_180)
                elif r == 270:
                    output = cv2.rotate(output, cv2.ROTATE_90_CLOCKWISE)
                
            ensemble += output

    ensemble /= 8.0
                        
    return ensemble    
#%%
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


model_gen1 = SwinIR(upscale=8, img_size=(48, 48), in_chans=3,
                window_size=8, img_range=1., depths=[6, 6, 6, 6, 6, 6],
                embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffle')
weights1 = torch.load('model_trained_x8_bst/SwinIR_1.pt')


model_gen1.load_state_dict(weights1)
model_gen1.to(device)

model_gen2 = SwinIR(upscale=8, img_size=(48, 48), in_chans=3,
                window_size=8, img_range=1., depths=[6, 6, 6, 6, 6, 6],
                embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffle')
weights2 = torch.load('model_trained_x8_bst/SwinIR_2.pt')


model_gen2.load_state_dict(weights2)
model_gen2.to(device)


model_gen3 = SwinIR(upscale=8, img_size=(48, 48), in_chans=3,
                window_size=8, img_range=1., depths=[6, 6, 6, 6, 6, 6],
                embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffle')
weights3 = torch.load('model_trained_x8_bst/SwinIR_3.pt')
model_gen3.load_state_dict(weights3)
model_gen3.to(device)


path = '../dataset_CIDIS_sisr_x8/thermal/test/sisr_x8/LR_x8'
img_list = sorted(glob.glob(path+'/*.bmp'))

save_path = 'test_result'
os.makedirs(save_path, exist_ok=True)

model_gen1.eval()
model_gen2.eval()
model_gen3.eval()

with torch.no_grad():
    for img_path in tqdm(img_list,desc='testing'):
        img_name = os.path.split(img_path)[-1]
        
        
        LR = cv2.imread(img_path, cv2.IMREAD_UNCHANGED) # cv2.IMREAD_GRAYSCALE, cv2.IMREAD_UNCHANGED        
        
        
        SR = GEOMETERY_ENSEMBLE(model_gen1,LR,8)
        SR += GEOMETERY_ENSEMBLE(model_gen2,LR,8)
        SR += GEOMETERY_ENSEMBLE(model_gen3,LR,8)
        SR /= 3
        
        cv2.imwrite(save_path+'/'+img_name,SR)

