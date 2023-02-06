#!/usr/bin/env python
# coding: utf-8

import os, sys

import matplotlib
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torchvision
import gc

from src import distributions
import torch.nn.functional as F

from src.resnet2 import ResNet_D
from src.unet import UNet

from src.tools import unfreeze, freeze
from src.tools import load_dataset, get_Z_pushed_loader_stats
from src.fid_score import calculate_frechet_distance
from src.tools import weights_init_D
from src.plotters import plot_random_Z_images, plot_Z_images

from copy import deepcopy
import json

# from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm

from IPython.display import clear_output

import wandb
from src.tools import fig2data, fig2img # for wandb

# This needed to use dataloaders for some datasets
from PIL import PngImagePlugin
LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)


# ## Main Config

DEVICE_IDS = [0]

DATASET1, DATASET1_PATH = 'handbag', 'data/mini_handbag_128.hdf5'
DATASET2, DATASET2_PATH = 'shoes', 'data/mini_shoes_128.hdf5'
# DATASET1, DATASET1_PATH = 'handbag', 'data/small_handbag_128.hdf5'
# DATASET2, DATASET2_PATH = 'shoes', 'data/small_shoes_128.hdf5'

# DATASET1, DATASET1_PATH = 'celeba_female', '../../data/img_align_celeba'
# DATASET2, DATASET2_PATH = 'aligned_anime_faces', '../../data/aligned_anime_faces'

T_ITERS = 10
D_LR, T_LR = 1e-4, 1e-4
IMG_SIZE = 64

ZC = 1
Z_STD = 0.1

BATCH_SIZE = 16
Z_SIZE = 8

PLOT_INTERVAL = 100
COST = 'weak_mse'
CPKT_INTERVAL = 2000
MAX_STEPS = 100001
SEED = 0x000000

GAMMA0, GAMMA1 = 0.0, 0.66
W_ITERS = 25000

EXP_NAME = f'{DATASET1}_{DATASET2}_T{T_ITERS}_{COST}_{IMG_SIZE}'
OUTPUT_PATH = './checkpoints/{}/{}_{}_{}/'.format(COST, DATASET1, DATASET2, IMG_SIZE)


# ## Preparation

config = dict(
    DATASET1=DATASET1,
    DATASET2=DATASET2, 
    T_ITERS=T_ITERS,
    D_LR=D_LR, T_LR=T_LR,
    BATCH_SIZE=BATCH_SIZE
)
    
assert torch.cuda.is_available()
torch.cuda.set_device(f'cuda:{DEVICE_IDS[0]}')
torch.manual_seed(SEED); np.random.seed(SEED)

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

# ## Prepare Samplers (X, Y)

X_sampler, X_test_sampler = load_dataset(DATASET1, DATASET1_PATH, img_size=IMG_SIZE)
Y_sampler, Y_test_sampler = load_dataset(DATASET2, DATASET2_PATH, img_size=IMG_SIZE)

torch.cuda.empty_cache(); gc.collect()
clear_output()


# # Initializing Networks

D = ResNet_D(IMG_SIZE, nc=3).cuda()
D.apply(weights_init_D)
print("yey")

T = UNet(3+ZC, 3, base_factor=48).cuda()  # ZC - noise input channels z

if len(DEVICE_IDS) > 1:
    T = nn.DataParallel(T, device_ids=DEVICE_IDS)
    D = nn.DataParallel(D, device_ids=DEVICE_IDS)

print('T params:', np.sum([np.prod(p.shape) for p in T.parameters()]))
print('D params:', np.sum([np.prod(p.shape) for p in D.parameters()]))


torch.manual_seed(0xDEADBEEF); np.random.seed(0xDEADBEEF)
X_fixed = X_sampler.sample(10)[:,None].repeat(1,4,1,1,1)
with torch.no_grad():
    Z_fixed = torch.randn(10, 4, ZC, IMG_SIZE, IMG_SIZE, device='cuda') * Z_STD
    XZ_fixed = torch.cat([X_fixed, Z_fixed], dim=2)
del X_fixed, Z_fixed
Y_fixed = Y_sampler.sample(10)

X_test_fixed = X_test_sampler.sample(10)[:,None].repeat(1,4,1,1,1)
with torch.no_grad():
    Z_test_fixed = torch.randn(10, 4, ZC, IMG_SIZE, IMG_SIZE, device='cuda') * Z_STD
    XZ_test_fixed = torch.cat([X_test_fixed, Z_test_fixed], dim=2)
del X_test_fixed, Z_test_fixed
Y_test_fixed = Y_test_sampler.sample(10)

# # Run Training

wandb.init(project="Handbag2Shoe", entity="amal-project", name=EXP_NAME, config=config)

T_opt = torch.optim.Adam(T.parameters(), lr=T_LR, weight_decay=1e-10)
D_opt = torch.optim.Adam(D.parameters(), lr=D_LR, weight_decay=1e-10)


for step in tqdm(range(MAX_STEPS)):
    gamma = min(GAMMA1, GAMMA0 + (GAMMA1-GAMMA0) * step / W_ITERS)
    # T optimization
    unfreeze(T); freeze(D)
    for t_iter in range(T_ITERS): 
        T_opt.zero_grad()
        X = X_sampler.sample(BATCH_SIZE)[:,None].repeat(1,Z_SIZE,1,1,1)
        with torch.no_grad():
            Z = torch.randn(BATCH_SIZE, Z_SIZE, ZC, IMG_SIZE, IMG_SIZE, device='cuda') * Z_STD
            XZ = torch.cat([X, Z], dim=2)
        T_XZ = T(
            XZ.flatten(start_dim=0, end_dim=1)
        ).permute(1,2,3,0).reshape(3, IMG_SIZE, IMG_SIZE, -1, Z_SIZE).permute(3,4,0,1,2)
        
        T_loss = F.mse_loss(X[:,0], T_XZ.mean(dim=1)).mean() - \
        D(T_XZ.flatten(start_dim=0, end_dim=1)).mean() + \
        T_XZ.var(dim=1).mean() * (1 - gamma - 1. / Z_SIZE)
        
        T_loss.backward(); T_opt.step()
    del T_loss, T_XZ, X, Z; gc.collect(); torch.cuda.empty_cache()

    # D optimization
    freeze(T); unfreeze(D)
    X = X_sampler.sample(BATCH_SIZE)
    with torch.no_grad():
        Z = torch.randn(BATCH_SIZE, ZC, X.size(2), X.size(3), device='cuda') * Z_STD
        XZ = torch.cat([X,Z], dim=1)
        T_XZ = T(XZ)
    Y = Y_sampler.sample(BATCH_SIZE)
    D_opt.zero_grad()
    D_loss = D(T_XZ).mean() - D(Y).mean()
    D_loss.backward(); D_opt.step();
    wandb.log({f'D_loss' : D_loss.item()}, step=step)
    del D_loss, Y, X, T_XZ, Z, XZ; gc.collect(); torch.cuda.empty_cache()
        
    if step % PLOT_INTERVAL == 0:
        clear_output(wait=True)
        print('Plotting ', step)
        
        fig, axes = plot_Z_images(XZ_fixed, Y_fixed, T)
        wandb.log({'Fixed Images' : [wandb.Image(fig2img(fig))]}, step=step)
        plt.show(fig); plt.close(fig) 
        
        fig, axes = plot_random_Z_images(X_sampler, ZC, Z_STD,  Y_sampler, T)
        wandb.log({'Random Images' : [wandb.Image(fig2img(fig))]}, step=step)
        plt.show(fig); plt.close(fig) 
        
        fig, axes = plot_Z_images(XZ_test_fixed, Y_test_fixed, T)
        wandb.log({'Fixed Test Images' : [wandb.Image(fig2img(fig))]}, step=step)
        plt.show(fig); plt.close(fig) 
        
        fig, axes = plot_random_Z_images(X_test_sampler, ZC, Z_STD,  Y_test_sampler, T)
        wandb.log({'Random Test Images' : [wandb.Image(fig2img(fig))]}, step=step)
        plt.show(fig); plt.close(fig) 

    if step % CPKT_INTERVAL == CPKT_INTERVAL - 1:
        freeze(T); 
        torch.save(T.state_dict(), os.path.join(OUTPUT_PATH, f'T_{SEED}_{step}.pt'))
        torch.save(D.state_dict(), os.path.join(OUTPUT_PATH, f'D_{SEED}_{step}.pt'))
        torch.save(D_opt.state_dict(), os.path.join(OUTPUT_PATH, f'D_opt_{SEED}_{step}.pt'))
        torch.save(T_opt.state_dict(), os.path.join(OUTPUT_PATH, f'T_opt_{SEED}_{step}.pt'))
    
    gc.collect(); torch.cuda.empty_cache()
