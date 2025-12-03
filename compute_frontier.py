# %% imports
import datetime
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn
import torch

from src.fractal import stability_test
from src.utils import get_freer_gpu

# %% setup
device = get_freer_gpu()
dtype = torch.float32
data_folder = "data/runs/"

# %% parameters
seed = 0
width = 100  # state size
depth = 1000  # number of layers
mode = "rand"  # in ['rand', 'struct', 'conv']
extra = ""  # additional name for saving
save = False

normalize = False  # layer normalization
n_channels = 1  # multiple networks and average errors
n_linops = depth  # number of linops to iterate on
resid_span = None  # residual connection length
resid_stride = None  # residual connection interval

stability_mode = "independent"  # in ['sensitivity', 'independent']
noise_level = 1e-5  # for sensitivity analysis
resolution = 1000  # number of scales

# struct
n_layers = 2
mags = ["unit", "unit"]  # in ['marchenko', 'unit']
osr = 1.01  # oversampling ratio

# conv
kernel_size = width

if mode == "rand":
    n_layers = None
    mags = None
    osr = None
    kernel_size = None

# Bounds for n_res = 100
# weight_scale_bounds = [0, 4]
# bias_scale_bounds = [0, 4]
# weight_scale_bounds = [2.0, 2.4]
# bias_scale_bounds = [1.8, 2.2]
# weight_scale_bounds = [2.15, 2.25]
# bias_scale_bounds = [2.0, 2.1]
weight_scale_bounds = [2.1875, 2.2125]
bias_scale_bounds = [2.0375, 2.0625]
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
save_folder = f"{timestamp}_{mode}_w{width}d{depth}{'_kernel' + str(kernel_size) if mode == 'conv' else ''}{'_layer' + str(n_layers) if mode == 'struct' else ''}_{extra}_seed{seed}_weight{weight_scale_bounds}_bias{bias_scale_bounds}/"

# %% compute frontier
metric_erf = stability_test(
    width=width,
    depth=depth,
    mode=mode,
    n_channels=n_channels,
    n_linops=n_linops,
    constant_input=False,
    normalize=normalize,
    residual_length=resid_span,
    residual_interval=resid_stride,
    n_layers=n_layers,
    mags=mags,
    osr=osr,
    kernel_size=kernel_size,
    stability_mode=stability_mode,
    noise_level=noise_level,
    resolution=resolution,
    weight_scale_bounds=weight_scale_bounds,
    bias_scale_bounds=bias_scale_bounds,
    dtype=dtype,
    device=device,
    seed=seed,
)

# %% plot and save
plt.figure()
seaborn.set_style("whitegrid")
metric_erf = metric_erf.cpu().numpy()
img = metric_erf.T
threshold = 1e-5
img[img < threshold] = threshold
bias_min = 0
bias_max = 1
weight_min = 0
weight_max = 1
plt.imshow(
    img[
        int(bias_min * resolution) : int(bias_max * resolution),
        int(weight_min * resolution) : int(weight_max * resolution),
    ],
    norm=matplotlib.colors.LogNorm(vmin=1e-10, vmax=1),
)  #

ax = plt.gca()
plt.grid(False)
plt.clim(threshold, 1)
plt.colorbar()

bias_scale_min = bias_scale_bounds[0] + bias_min * (
    bias_scale_bounds[1] - bias_scale_bounds[0]
)
bias_scale_max = bias_scale_bounds[0] + bias_max * (
    bias_scale_bounds[1] - bias_scale_bounds[0]
)
weight_scale_min = weight_scale_bounds[0] + weight_min * (
    weight_scale_bounds[1] - weight_scale_bounds[0]
)
weight_scale_max = weight_scale_bounds[0] + weight_max * (
    weight_scale_bounds[1] - weight_scale_bounds[0]
)
ylab = np.linspace(bias_scale_min, bias_scale_max, num=int(bias_scale_bounds[1] + 1))
xlab = np.linspace(
    weight_scale_min, weight_scale_max, num=int(weight_scale_bounds[1] + 1)
)
indXx = np.linspace(0, resolution - 1, num=xlab.shape[0]).astype(int)
indXy = np.linspace(0, resolution - 1, num=ylab.shape[0]).astype(int)

ax.set_xticks(indXx)
ax.set_xticklabels(xlab)
ax.set_yticks(indXy)
ax.set_yticklabels(ylab)
ax.set_xlabel("Weight variance")
ax.set_ylabel("Bias variance")

if save is True:
    if not os.path.exists(data_folder + save_folder):
        os.makedirs(data_folder + save_folder)
    np.save(data_folder + save_folder + "metric_erf.npy", metric_erf)
    np.save(data_folder + save_folder + "xlab.npy", xlab)
    np.save(data_folder + save_folder + "ylab.npy", ylab)
    plt.savefig(data_folder + save_folder + "frontier.pdf")

plt.show()
