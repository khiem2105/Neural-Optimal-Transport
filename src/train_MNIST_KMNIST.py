import torch
import numpy as np
torch.manual_seed(0xBADBEEF); np.random.seed(0xBADBEEF)

from datasets import load_digit_dataset
from trainer import Trainer
from config import Config

import wandb
wandb.login()

def run():
    config = Config(
        img_size=32,
        img_c_in=1,
        img_c_out=1,
        batch_size=64,
        z_size=4,
        gamma_0=0.01,
        gamma_1=1.0,
        gamma_iters=25000,
        T_iters=10,
        plot_interval=100,
        ckpt_interval=2000,
        max_steps=40001
    )

    X_train_sampler, X_test_sampler = load_digit_dataset(
        batch_size=config.batch_size,
        img_size=config.img_size,
        name="MNIST",
        root="datasets"
    )
    Y_train_sampler, Y_test_sampler = load_digit_dataset(
        batch_size=config.batch_size,
        img_size=config.img_size,
        name="KMNIST",
        root="datasets"
    )

    X_fixed = X_train_sampler.sample(10)[:,None].repeat(1,4,1,1,1)
    with torch.no_grad():
        Z_fixed = torch.randn(10, 4, config.zc, config.img_size, config.img_size, device='cuda') * config.z_std
        XZ_fixed = torch.cat([X_fixed, Z_fixed], dim=2)
    del X_fixed, Z_fixed
    Y_fixed = Y_train_sampler.sample(10)

    X_test_fixed = X_test_sampler.sample(10)[:,None].repeat(1,4,1,1,1)
    with torch.no_grad():
        Z_test_fixed = torch.randn(10, 4, config.zc, config.img_size, config.img_size, device='cuda') * config.z_std
        XZ_test_fixed = torch.cat([X_test_fixed, Z_test_fixed], dim=2)
    del X_test_fixed, Z_test_fixed
    Y_test_fixed = Y_test_sampler.sample(10)

    dataset_1 = "MNIST"
    dataset_2 = "KMNIST"
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
        max_steps=config.max_steps
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

