import datetime
import os
from warnings import warn

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn
import torch

from src.fractal import stability_test


def get_freer_gpu(verbose=True):
    """
    Returns the GPU device with the most free memory.

    Use in conjunction with ``torch.cuda.is_available()``.
    Attempts to use ``nvidia-smi`` with ``bash``, if these don't exist then uses torch commands to get free memory.

    :param bool verbose: print selected GPU index and memory
    :return torch.device device: selected torch cuda device.
    """
    try:
        os.system(
            "nvidia-smi -q -d Memory |grep -A5 GPU|grep Free >tmp"
            if os.name == "posix"
            else 'bash -c "nvidia-smi -q -d Memory |grep -A5 GPU|grep Free >tmp"'
        )
        memory_available = [int(x.split()[2]) for x in open("tmp", "r").readlines()]
        idx, mem = np.argmax(memory_available), np.max(memory_available)
        device = torch.device(f"cuda:{idx}")

    except Exception:
        if torch.cuda.device_count() == 0:
            warn("Couldn't find free GPU")
            return torch.device("cuda")

        else:
            # Note this is slower and will return slightly different values to nvidia-smi
            idx, mem = max(
                (
                    (d, torch.cuda.mem_get_info(d)[0] / 1048576)
                    for d in range(torch.cuda.device_count())
                ),
                key=lambda x: x[1],
            )
            device = torch.device(f"cuda:{idx}")

    if verbose:
        print(f"Selected GPU {idx} with {mem} MiB free memory ")

    return device


device = get_freer_gpu()
data_folder = "data/runs/"

seed = 0
width = 20  # state size
depth = 1000  # input length for reservoir
mode = "rand"  # in ['rand', 'struct', 'conv']
additional = ""  # additional name for saving

normalize = False  # layer normalization
n_channels = 1  # multiple networks and average errors
n_linops = depth  # number of linops to iterate on
residual_length = None  # residual connection length
residual_interval = None  # residual connection interval

stability_mode = "independent"  # in ['sensitivity', 'independent']
noise_level = 1e-5  # for sensitivity analysis
resolution = 1000  # number of scales

# struct
n_layers = 1.5
mags = ["unit"]  # in ['marchenko', 'unit']
osr = 1.01  # oversampling ratio

# conv
kernel_size = width

if mode == "rand":
    n_layers = None
    mags = None
    osr = None
    kernel_size = None

save = True

# Bounds for n_res = 100
weight_scale_bounds = [0, 4]
bias_scale_bounds = [0, 4]
# weight_scale_bounds = [2.0, 2.4]
# bias_scale_bounds = [1.8, 2.2]
# weight_scale_bounds = [2.15, 2.25]
# bias_scale_bounds = [2.0, 2.1]
# weight_scale_bounds = [2.1875, 2.2125]
# bias_scale_bounds = [2.0375, 2.0625]
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
save_folder = f"{timestamp}_{mode}_w{width}d{depth}{'_kernel' + str(kernel_size) if mode == 'conv' else ''}{'_layer' + str(n_layers) if mode == 'struct' else ''}_{additional}_seed{seed}_weight{weight_scale_bounds}_bias{bias_scale_bounds}/"

metric_erf = stability_test(
    width=width,
    depth=depth,
    mode=mode,
    n_channels=n_channels,
    n_linops=n_linops,
    constant_input=False,
    normalize=normalize,
    residual_length=residual_length,
    residual_interval=residual_interval,
    n_layers=n_layers,
    mags=mags,
    osr=osr,
    kernel_size=kernel_size,
    stability_mode=stability_mode,
    noise_level=noise_level,
    resolution=resolution,
    weight_scale_bounds=weight_scale_bounds,
    bias_scale_bounds=bias_scale_bounds,
    device=device,
    seed=seed,
)

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
