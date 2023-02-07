import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from resnet2 import ResNet_D
from unet import UNet
from tools import weights_init_D, freeze, unfreeze, fig2img
from distributions import LoaderSampler
from plotters import plot_Z_images, plot_random_Z_images

import wandb

import os
from tqdm import tqdm
import gc

SEED = 0xBADBEEF
torch.manual_seed(SEED)
np.random.seed(SEED)

class Trainer():
    def __init__(
        self,
        img_size: int,
        batch_size: int,
        zc: int,
        z_std: float,
        z_size: int,
        T_iters: int,
        D_lr: float,
        T_lr: float,
        gamma_0: float,
        gamma_1: float,
        gamma_iters: int,
        save_path: str,
        img_c_in: int=3,
        img_c_out: int=3,
        base_factor: int=48,
        plot_interval: int=1000,
        ckpt_interval: int=1000,
        ckpt_path: str=None,
        cost: str="weak",
        max_steps: int=100001        
    ):
        super(Trainer, self).__init__()

        # Set device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Load checkpoint if there is
        if ckpt_path:
            self.ckpt = torch.load(ckpt_path, map_location=self.device)
        else:
            self.ckpt = None

        # Initialize T network
        if cost == "weak":
            self.T = UNet(
                n_channels_in=img_c_in+zc,
                n_channels_out=img_c_out,
                base_factor=base_factor
            ).to(self.device)
        else:
            self.T = UNet(
                n_channels_in=img_c_in,
                n_channels_out=img_c_out,
                base_factor=base_factor
            ).to(self.device)
        
        # Initialize D network
        self.D = ResNet_D(
            size=img_size,
            nc=img_c_out
        ).to(self.device)

        # Initialize T opt
        self.T_opt = torch.optim.Adam(
            self.T.parameters(),
            lr=T_lr,
            weight_decay=1e-10
        )
        
        # Initialize D opt
        self.D_opt = torch.optim.Adam(
            self.D.parameters(),
            lr=D_lr,
            weight_decay=1e-10
        )

        # Load models and optimizers state dict from ckpt if there is
        if self.ckpt:
            self.T.load_state_dict(self.ckpt["T"])
            self.D.load_state_dict(self.ckpt["D"])
            self.T_opt.load_state_dict(self.ckpt["T_opt"])
            self.D_opt.load_state_dict(self.ckpt["D_opt"])
            self.curr_step = self.ckpt["curr_step"]
        else:
            self.D.apply(weights_init_D)
            self.curr_step = 0

        self.img_size = img_size
        self.batch_size = batch_size
        self.zc = zc
        self.z_std = z_std
        self.z_size = z_size
        self.T_iters = T_iters
        self.gamma_0 = gamma_0
        self.gamma_1 = gamma_1
        self.gamma_iters = gamma_iters
        self.img_c_in = img_c_in
        self.img_c_out = img_c_out
        self.base_factor = base_factor
        self.plot_interval = plot_interval
        self.ckpt_interval = ckpt_interval
        self.cost =  cost
        self.max_steps = max_steps
        self.save_path = save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    
    def train(
        self,
        X_train_sampler: LoaderSampler,
        X_test_sampler: LoaderSampler,
        Y_train_sampler: LoaderSampler,
        Y_test_sampler: LoaderSampler,
        XZ_fixed: torch.Tensor,
        XZ_test_fixed: torch.Tensor,
        Y_fixed: torch.Tensor,
        Y_test_fixed: torch.Tensor,
        logger
    ):

        for step in tqdm(range(self.curr_step, self.max_steps)):
            # Adaptive gamma
            gamma = min(self.gamma_1, self.gamma_0 + (self.gamma_1 - self.gamma_0) * step / self.gamma_iters)
            logger.log({"gamma": gamma}, step=step)

            # Update T
            unfreeze(self.T)
            freeze(self.D)

            for t_iter in range(self.T_iters):
                if self.cost == "weak":
                    X = X_train_sampler.sample(self.batch_size)[:, None].repeat(1, self.z_size, 1, 1, 1)
                    with torch.no_grad():
                        Z = torch.randn(self.batch_size, self.z_size, self.zc, self.img_size, self.img_size, device=self.device) * self.z_std
                        XZ = torch.cat([X, Z], dim=2)
                    
                    T_XZ = self.T(
                        XZ.flatten(start_dim=0, end_dim=1)
                    ).reshape(self.batch_size, -1, self.img_c_out, self.img_size, self.img_size)

                    T_loss = F.mse_loss(X, T_XZ) - T_XZ.var(dim=1).mean() * gamma - \
                             self.D(T_XZ.flatten(start_dim=0, end_dim=1)).mean()

                else:
                    X = X_train_sampler.sample(self.batch_size)
                    T_X = self.T(X)
                    
                    T_loss = F.mse_loss(X, T_X) - self.D(T_X).mean()

                logger.log({"T_loss": T_loss.item()}, step=step)
                
                self.T_opt.zero_grad()
                T_loss.backward()
                self.T_opt.step()
            
            if self.cost == "weak":
                del T_loss, X, Z, XZ, T_XZ
            else:
                del T_loss, X, T_X
            
            gc.collect()
            torch.cuda.empty_cache()
            
            # Update D
            freeze(self.T)
            unfreeze(self.D)

            X = X_train_sampler.sample(self.batch_size)
            Y = Y_train_sampler.sample(self.batch_size)
            with torch.no_grad():
                if self.cost == "weak":
                    Z = torch.randn(self.batch_size, self.zc, self.img_size, self.img_size, device=self.device) * self.z_std
                    XZ = torch.cat([X, Z], dim=1)
                    T_XZ = self.T(XZ)
                else:
                    T_XZ = self.T(X)
            
            D_loss = self.D(T_XZ).mean() - self.D(Y).mean()
            logger.log({"D_loss": D_loss.item()}, step=step)
            self.D_opt.zero_grad()
            D_loss.backward()
            self.D_opt.step()

            if self.cost == "weak":
                del D_loss, X, Z, XZ, T_XZ, Y
            else:
                del D_loss, X, T_XZ, Y
            
            gc.collect()
            torch.cuda.empty_cache()

            if step % self.ckpt_interval == self.ckpt_interval - 1 or step == self.max_steps - 1:
                ckpt = {
                    "T": self.T.state_dict(),
                    "D": self.D.state_dict(),
                    "T_opt": self.T_opt.state_dict(),
                    "D_opt": self.D_opt.state_dict(),
                    "curr_step": step
                }

                torch.save(ckpt, os.path.join(self.save_path, f"step={step}.pt"))
            
            if step % self.plot_interval == self.plot_interval - 1 or step == self.max_steps - 1:
                fig, axes = plot_Z_images(XZ_fixed, Y_fixed, self.T)
                logger.log({'Fixed Images' : [wandb.Image(fig2img(fig))]}, step=step)
                plt.close(fig)  
                
                fig, axes = plot_random_Z_images(X_train_sampler, self.zc, self.z_std,  Y_train_sampler, self.T)
                logger.log({'Random Images' : [wandb.Image(fig2img(fig))]}, step=step)
                plt.close(fig) 
                
                fig, axes = plot_Z_images(XZ_test_fixed, Y_test_fixed, self.T)
                logger.log({'Fixed Test Images' : [wandb.Image(fig2img(fig))]}, step=step) 
                plt.close(fig)

                fig, axes = plot_random_Z_images(X_test_sampler, self.zc, self.z_std,  Y_test_sampler, self.T)
                logger.log({'Random Test Images' : [wandb.Image(fig2img(fig))]}, step=step) 
                plt.close(fig)




