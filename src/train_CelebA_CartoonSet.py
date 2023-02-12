import torch
import numpy as np
torch.manual_seed(0xBADBEEF); np.random.seed(0xBADBEEF)

from tools import load_dataset
from datasets import load_celeba
from trainer import Trainer
from config import Config

import wandb
wandb.login()

def run():
    config = Config(
        img_size=128,
        img_c_in=3,
        img_c_out=3,
        batch_size=16,
        z_size=2,
        gamma_0=0.1,
        gamma_1=1.0,
        gamma_iters=10000,
        T_iters=10,
        plot_interval=200,
        ckpt_interval=2000,
        max_steps=40001
    )

    X_train_sampler, X_test_sampler = load_celeba(
        img_size=config.img_size,
        batch_size=config.batch_size,
        root="/tempory/NOT_datasets/",
        split_male_female=False
    )
    
    Y_train_sampler, Y_test_sampler = load_dataset(
        name="cartoon",
        path="/tempory/NOT_datasets/cartoonset100k_jpg",
        img_size=config.img_size,
        batch_size=config.batch_size
    )

    X_fixed = X_train_sampler.sample(10)[:,None].repeat(1,4,1,1,1)
    with torch.no_grad():
        Z_fixed = torch.rand(10, 4, config.zc, config.img_size, config.img_size, device='cuda')
        XZ_fixed = torch.cat([X_fixed, Z_fixed], dim=2)
    del X_fixed, Z_fixed
    Y_fixed = Y_train_sampler.sample(10)

    X_test_fixed = X_test_sampler.sample(10)[:,None].repeat(1,4,1,1,1)
    with torch.no_grad():
        Z_test_fixed = torch.rand(10, 4, config.zc, config.img_size, config.img_size, device='cuda')
        XZ_test_fixed = torch.cat([X_test_fixed, Z_test_fixed], dim=2)
    del X_test_fixed, Z_test_fixed
    Y_test_fixed = Y_test_sampler.sample(10)

    dataset_1 = "CelebA"
    dataset_2 = "Cartoon Set"
    run_name = f"{dataset_1}_{dataset_2}_{config.cost}_gamma={config.gamma_1}_T_iters={config.T_iters}_z_size={config.z_size}"

    run = wandb.init(
        project="Neural_Optimal_Transport",
        name=run_name,
        config=config.__dict__
    )

    save_path = f"../checkpoints/{dataset_1}_{dataset_2}/{config.cost}_gamma={config.gamma_1}_T_iter={config.T_iters}_z_size={config.z_size}"

    trainer = Trainer(
        img_size=config.img_size,
        batch_size=config.batch_size,
        zc=config.zc,
        z_std=config.z_std,
        z_size=config.z_size,
        T_iters=config.T_iters,
        D_lr=config.D_lr,
        T_lr=config.T_lr,
        gamma_0=config.gamma_0,
        gamma_1=config.gamma_1,
        gamma_iters=config.gamma_iters,
        save_path=save_path,
        img_c_in=config.img_c_in,
        img_c_out=config.img_c_out,
        base_factor=config.base_factor,
        plot_interval=config.plot_interval,
        ckpt_interval=config.ckpt_interval,
        cost=config.cost,
        max_steps=config.max_steps,
        z_sampler="uniform"
    )

    trainer.train(
        X_train_sampler,
        X_test_sampler,
        Y_train_sampler,
        Y_test_sampler,
        XZ_fixed,
        XZ_test_fixed,
        Y_fixed,
        Y_test_fixed,
        run
    )

    run.finish()

if __name__ == "__main__":
    run()

