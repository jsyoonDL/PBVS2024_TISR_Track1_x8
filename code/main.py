# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 10:56:52 2024

@author: user
"""

import time
import matplotlib.pyplot as plt
from torch import optim
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.autograd import Variable
import numpy as np
import random
import torch.multiprocessing

from Discriminator import Discriminator
from SwinIR import SwinIR
from Dataset import *
from loss import *
from pytorch_msssim import ssim


#%%
def set_seed(seed = 0):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    # Set a fixed value for the hash seedTrue
    os.environ['PYTHONHASHSEED'] = str(seed)
#%%
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
#%%
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count 

#%% train
def train(model_gen, model_dis,params):
    

    # load params
    img_size = params['img_size']
    num_epochs = params['num_epochs']
    beta1=params['beta1']
    beta2=params['beta2']
    path2img = params['path2img']    
    transform = params['transform']
    batch_size = params['batch_size']
    
    lr = params['learning_rate']
    lambda_vals = params['lambda_vals']
        
    path2models = params['path2models']
    
    lambda_gan, lambda_pixel, lambda_pv= lambda_vals # loss weight
     
    #%% reproducibility
    seed=1234
    g = torch.Generator()
    g.manual_seed(seed)
    set_seed(seed)

    
    #%% load dataset (tr. & val.)
    train_ds = DatasetTrain(path2img['tr'], transform=transform)
    train_dl = DataLoader(train_ds, batch_size=batch_size,
                          shuffle=True,
                          num_workers=0,
                          worker_init_fn=seed_worker,
                          generator=g)
    
    test_ds = Dataset(path2img['te'], transform=transform)
    test_dl = DataLoader(test_ds, 
                         batch_size=12, 
                         shuffle=True, 
                         num_workers=0)
    
    #%% optimization
    opt_gen = optim.AdamW(model_gen.parameters(),lr=lr,betas=(beta1,beta2))    
    opt_model_dis = optim.AdamW(model_dis.parameters(),lr=lr,betas=(beta1,beta2))

    
    scheduler = CosineAnnealingLR(opt_gen, T_max=len(train_dl), eta_min=1e-10, last_epoch=-1)
    scheduler_D = CosineAnnealingLR(opt_model_dis, T_max=len(train_dl), eta_min=1e-10, last_epoch=-1)
    
    #%% loss

    loss_func_pix = CharbonnierLoss().to(device)

    loss_func_dis = nn.BCEWithLogitsLoss().to(device)

    psnr = Loss_PSNR().to(device)
    loss_func_mse = nn.MSELoss().to(device)
    loss_func_l1 = nn.L1Loss().to(device)
    
    
    loss_hist = {'tr':[],
                 'val':[],
                 }
    
    loss_hist = {'gen':[],
                 'dis':[],
                 'val':[]
                 }
    batch_count = params['batch_count']
    tot_g_loss = AverageMeter()
    tot_d_loss = AverageMeter()
    gan = AverageMeter()

    loss_func_ploss = VGGLoss(device,gray=False).to(device)
    #%%
    patch_h, patch_w = int(img_size[0] / 2 ** 4), int(img_size[1] / 2 ** 4)
    init_chk_loss = 1000
    noise_factor = 1e-3
    #%% training


    for epoch in range(num_epochs):
        model_gen.train()
        model_dis.train()
        
        with tqdm(train_dl, unit="batch") as tepoch:
            
            for LR, HR in tepoch:
                tepoch.set_description(f"LR {opt_gen.param_groups[0]['lr']},Epoch {epoch}") # progress bar
                
       
                # real image
                LR = Variable(LR.to(device))
                HR = Variable(HR.to(device))
                
    
                
                valid = Variable(torch.ones(LR.size(0),1,patch_h,patch_w, requires_grad=False).to(device)-noise_factor*torch.rand(LR.size(0),1,patch_h,patch_w, requires_grad=False).to(device))
                fake = Variable(torch.zeros(LR.size(0),1,patch_h,patch_w, requires_grad=False).to(device)+noise_factor*torch.rand(LR.size(0),1,patch_h,patch_w, requires_grad=False).to(device))
               
                # -----------------------
                #  Train generator
                # -----------------------
                
                opt_gen.zero_grad()
                
                SR = model_gen(LR) # gen. fake images
                SR = torch.clamp(SR,0,1)
                
                # pixel loss                
                   
                pixel_loss = 0.99*loss_func_pix(SR,HR)
                pixel_loss +=0.01*(1-ssim(SR, HR, data_range=1.0, size_average=True)) # return (N,)
               
                
                # perceptual loss 
                pv_loss = loss_func_ploss(SR, HR) 
              
                
                # GAN loss
                loss_GAN = loss_func_dis(model_dis(SR), valid)
                gan.update(loss_GAN.item())
                ###            
                
                g_loss = lambda_gan*loss_GAN + lambda_pixel*pixel_loss + lambda_pv*pv_loss
                
                g_loss.backward()
                
                opt_gen.step()
                scheduler.step() # iter

                
                tot_g_loss.update(g_loss.item())
        
                loss_hist['gen'].append(g_loss.item())
                
                
                # -----------------------
                #  Train Discriminator
                # -----------------------

                opt_model_dis.zero_grad()

                # Loss of real and fake images
                loss_real = loss_func_dis(model_dis(HR), valid)
                loss_fake = loss_func_dis(model_dis(SR.detach()), fake)
                
                                    
        
                # Total loss
                loss_D = (loss_real + loss_fake) / 2          
                
           
                
                
                loss_D.backward()
                opt_model_dis.step()
                scheduler_D.step() # iter  
                
                # -----------------------
                #  Validation
                # -----------------------
                
                tot_d_loss.update(loss_D.item())
                loss_hist['dis'].append(loss_D.item())
                
                
                i=0
                if batch_count % 1000 == 0:   
                    model_gen.eval()
                    tot_val_loss = AverageMeter()
                    with torch.no_grad():
                        for LR_t,HR_t in test_dl:                            
  
                            LR_t = Variable(LR_t.to(device))
                            HR_t = Variable(HR_t.to(device))
                            
                            SR_t = model_gen(LR_t)
                            SR_t = torch.clamp(SR_t,0,1)
                            
                            if i ==0:
                                SR_t_show = SR_t.detach().cpu()
                                HR_t_show = HR_t
                                LR_t_show = LR_t
                        

                            v_loss = loss_func_mse(SR_t, HR_t)
                            
                            i+=1
                            tot_val_loss.update(v_loss.item())
                    
                    loss_hist['val'].append(tot_val_loss.avg)
                    if init_chk_loss > tot_val_loss.avg:
                        os.makedirs(path2models+'/gen_bst', exist_ok=True) 
                        os.makedirs(path2models+'/dis_bst', exist_ok=True)
                        torch.save(model_gen.state_dict(), os.path.join(path2models, 'gen_bst/weights_gen_SR_bst'+str(batch_count)+'.pt'))       
                        torch.save(model_dis.state_dict(), os.path.join(path2models, 'dis_bst/weights_dis_bst'+str(batch_count)+'.pt')) 
                        
                        init_chk_loss = tot_val_loss.avg
                        
                    if batch_count % 5000 == 0: #check point save
                        os.makedirs(path2models+'/gen_chk', exist_ok=True) 
                        os.makedirs(path2models+'/dis_chk', exist_ok=True)
                        torch.save(model_gen.state_dict(), os.path.join(path2models, 'gen_chk/weights_gen_SR_chk_pnt'+str(batch_count)+'.pt'))       
                        torch.save(model_dis.state_dict(), os.path.join(path2models, 'dis_chk/weights_dis_chk_pnt'+str(batch_count)+'.pt'))  
                        noise_factor = noise_factor*.5 # noise weigts degradation
                    
                    
                    # visualization   
                    fig = plt.figure(0,figsize=(10,10))       

                    idx = 0
                    for ii in range(0,3):
                        plt.subplot(3,3,3*ii+1)
                        plt.imshow(to_pil_image(LR_t_show[ii]),cmap = 'gray')
                        plt.axis('off')
                        if ii<1: plt.title('Input')
                        plt.subplot(3,3,3*ii+2)
                        plt.imshow(to_pil_image(SR_t_show[ii]),cmap = 'gray')
                        plt.axis('off')        
                        if ii<1: plt.title('x8')
                        plt.subplot(3,3,3*ii+3)
                        plt.imshow(to_pil_image(HR_t_show[ii]),cmap = 'gray')
                        plt.axis('off')
                        if ii<1: plt.title('GT')
                        
                    fig.tight_layout()     
 
                    fig_save_path = 'intermidiate_result_x8_step1'
                    os.makedirs(fig_save_path, exist_ok=True)
                    
                    fig.savefig(fig_save_path +'/Itr_'+str(batch_count)+'.png')
                    plt.close('all')
                batch_count += 1
                
                # diplay
                tepoch.set_postfix(GAN_loss = gan.avg, g_loss=tot_g_loss.avg,d_loss= tot_d_loss.avg, 
                                   v_loss = tot_val_loss.avg, v_loss_b = init_chk_loss)
                
                
    return (model_gen,model_dis,loss_hist,batch_count)
#%% model gen
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


model_gen = SwinIR(upscale=8, in_chans=3, img_size=(48,48), window_size=8,
                img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')
param_key_g = 'params'

pretrained_model = torch.load('TrainedModels/SwinIRs48w8_x8/001_classicalSR_DIV2K_s48w8_SwinIR-M_x8.pth')
model_gen.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)
model_gen.to(device)

path2models = 'model_trained_x8_step1'


#%% def. discriminator    
model_dis = Discriminator(input_shape=(3,384,384))

model_dis.to(device)


#%% params
transform = transforms.Compose([                                 
                                transforms.ToTensor()
                                ])
batch_count =0

params = {
    'num_epochs': 300000,
    'learning_rate':2e-4,# 
    'beta1':0.99,
    'beta2':0.999,
    'batch_size':14, # 12
    'lambda_vals': [1e-6,1,1e-6], # 1e-3,1,1e-3
    'path2img': {'tr':'../dataset_CIDIS_sisr_x8/thermal/train/GT','te':'../dataset_CIDIS_sisr_x8/thermal/val/LR_x8'},
    'path2models': path2models,
    'device':device,
    'transform': transform,
    'batch_count': batch_count,
    'img_size': [384,384],
}

#%% output with time complexity  

in_time = time.time() 

model_gen,model_dis,loss_hist,batch_count = train(model_gen,model_dis,params)

path2weights_gen = os.path.join(path2models, 'weights_gen_SR_last.pt')
torch.save(model_get.state_dict(), path2weights_gen)

path2weights_dis = os.path.join(path2models, 'weights_dis_last.pt')
torch.save(model_dis.state_dict(), path2weights_dis)
 
out_time = time.time()
pro_time = out_time-in_time 
print(pro_time)
