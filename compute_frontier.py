# %% imports
import datetime
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn
import torch

from src.fractal import stability_test
from src.utils import get_freer_gpu, plot_frontier

# %% setup
device = get_freer_gpu()
dtype = torch.float32
data_folder = "data/runs/"

# %% parameters
seed = 1
width = 100  # state size
depth = 1000  # number of layers
mode = "struct"  # in ['rand', 'struct', 'conv']
extra = ""  # additional name for saving
save = False

normalize = False  # layer normalization
n_linops = depth  # number of linops to iterate on
resid_span = None  # residual connection length
resid_stride = None  # residual connection interval

stability_mode = "independent"  # in ['sensitivity', 'independent']
noise_level = 1e-5  # for sensitivity analysis

# struct
n_layers = 2
mags = ["unit", "unit"]  # in ['marchenko', 'unit']
osr = 1.01  # oversampling ratio

# conv
kernel_size = 100

config_linop = {
    "n_linops": n_linops,
    "n_layers": n_layers,
    "mags": mags,
    "osr": osr,
    "kernel_size": kernel_size,
}
config_resid = {
    "resid_span": resid_span,
    "resid_stride": resid_stride,
}

if mode == "rand":
    n_layers = None
    mags = None
    osr = None
    kernel_size = None

resolution = [1000, 1000]  # number of weight and bias scales
chunks = [1, 1]
n_save_last = 1
# Bounds for n_res = 100
# W_scale_bounds = [0, 4]
# b_scale_bounds = [0, 4]
# * for convergence plot
# W_scale_bounds = [1.0, 2.5]
# b_scale_bounds = [1.0, 1.0]
# weight_scale_bounds = [2.0, 2.4]
# bias_scale_bounds = [1.8, 2.2]
# weight_scale_bounds = [2.15, 2.25]
# bias_scale_bounds = [2.0, 2.1]
W_scale_bounds = [2.1875, 2.2125]
b_scale_bounds = [2.0375, 2.0625]
# weight_scale_bounds = [2.1625, 2.1875]
# bias_scale_bounds = [2.0375, 2.0625]
# weight_scale_bounds = [2.168, 2.193]
# bias_scale_bounds = [2.0375, 2.0625]
# weight_scale_bounds = [2.3, 2.5]
# bias_scale_bounds = [2.3, 2.5]
# weight_scale_bounds = [1.8, 2.2]
# bias_scale_bounds = [2.0, 2.4]
# weight_scale_bounds = [3, 4]
# bias_scale_bounds = [3, 4]
# weight_scale_bounds = [3.75, 4.0]
# bias_scale_bounds = [3.25, 3.5]
# weight_scale_bounds = [3.75, 3.8]
# bias_scale_bounds = [3.25, 3.3]
# weight_scale_bounds = [1.51, 1.53]
# bias_scale_bounds = [0.96, 0.98]
# weight_scale_bounds = [1.4, 1.6]
# bias_scale_bounds = [0.6, 0.8]
# weight_scale_bounds = [1.5, 1.55]
# bias_scale_bounds = [0.65, 0.7]
# weight_scale_bounds = [1.62, 1.92]
# bias_scale_bounds = [1, 1.3]
# weight_scale_bounds = [1.73, 1.83]
# bias_scale_bounds = [1.1, 1.2]
# weight_scale_bounds = [1.777, 1.802]
# bias_scale_bounds = [1.137, 1.162]
# weight_scale_bounds = [1.785, 1.795]
# bias_scale_bounds = [1.145, 1.155]
# weight_scale_bounds = [1.70, 1.75]
# bias_scale_bounds = [0.25, 0.30]
# Bounds for n_res = 30
# weight_scale_bounds = [0, 2]
# bias_scale_bounds = [0, 2]
# weight_scale_bounds = [1.75, 2.05]
# bias_scale_bounds = [1, 1.3]
# weight_scale_bounds = [1.87, 1.97]
# bias_scale_bounds = [1.1, 1.2]
# get current date
timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
save_folder = f"{timestamp}_{mode}_w{width}d{depth}{'_kernel' + str(kernel_size) if mode == 'conv' else ''}{'_layer' + str(n_layers) if mode == 'struct' else ''}_{extra}_seed{seed}_weight{W_scale_bounds}_bias{b_scale_bounds}/"

# %% compute frontier
errs = stability_test(
    width=width,
    depth=depth,
    mode=mode,
    W_scale_bounds=W_scale_bounds,
    b_scale_bounds=b_scale_bounds,
    resolution=resolution,
    config_linop=config_linop,
    config_resid=config_resid,
    constant_bias=False,
    normalize=normalize,
    stability_mode=stability_mode,
    noise_level=noise_level,
    n_save_last=n_save_last,
    chunks=chunks,
    dtype=dtype,
    device=device,
    seed=seed,
)  # returns shape (n_save_last, n_W_scales, n_b_scales)

# %%
errs.shape
# %% plot frontier and save
err = errs[-1]
plot_frontier(
    err.cpu().numpy(),
    W_scale_bounds,
    b_scale_bounds,
    resolution,
    save_path=None if not save else data_folder + save_folder,
)

# %% plot convergence
# errs should have shape (n_save_last, n_W_scales, 1)
# (only one bias scale)
plt.plot(errs[:, :, 0].cpu().numpy(), "-")
plt.yscale("log")
