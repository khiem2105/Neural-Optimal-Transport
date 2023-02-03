from dataclasses import dataclass

@dataclass
class Config:
    img_size: int
    batch_size: int
    z_size: int
    gamma_1: float
    gamma_iters: int
    gamma_0: float=0
    img_c_int: int=3
    img_c_out: int=3
    zc: int=1
    z_std: float=0.1
    T_iters: int=10
    T_lr: float=1e-4
    D_lr: float=1e-4
    base_factor: int=48
    plot_interval: int=2000
    ckpt_interval: int=2000
    cost: str="weak"
    max_steps: int=40001