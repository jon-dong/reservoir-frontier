import datetime
import os
from warnings import warn

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn
import torch

from utils import stability_test

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

    except:
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

seed = 1
width = 100 # state size
depth = 1000 # input length for reservoir
mode = "rand" # in ['rand', 'struct_rand', 'random_conv']
additional = '_sensi_10layer_' # additional name for saving

normalize = False # layer normalization
n_channels = 1 # multiple networks and average errors
n_linops = 1 # number of linops to iterate on
residual_length = None # residual connection length
residual_interval = None # residual connection interval

stability_mode = "sensitivity" # in ['sensitivity', 'independent']
noise_level = 1e-15 # for sensitivity analysis
resolution = 1000 # number of scales

# struct rand
n_layers = 1.5
mags = ["marchenko"] # in ['marchenko', 'unit']
osr = 1000 # oversampling ratio
# rand conv
kernel_size = 100

if mode == 'rand':
    n_layers = None
    mags = None
    osr = None

save = True

# Bounds for n_res = 100
# res_scale_bounds = [0, 2]
# input_scale_bounds = [0, 2]
weight_scale_bounds = [0, 4]
bias_scale_bounds = [0, 4]
# res_scale_bounds = [2.3, 2.5]
# input_scale_bounds = [2.3, 2.5]
# res_scale_bounds = [1.8, 2.2]
# input_scale_bounds = [2.0, 2.4]
# res_scale_bounds = [3, 4]
# input_scale_bounds = [3, 4]
# res_scale_bounds = [3.75, 4.0]
# input_scale_bounds = [3.25, 3.5]
# res_scale_bounds = [3.75, 3.8]
# input_scale_bounds = [3.25, 3.3]
# res_scale_bounds = [1.51, 1.53]
# input_scale_bounds = [0.96, 0.98]
# res_scale_bounds = [1.4, 1.6]
# input_scale_bounds = [0.6, 0.8]
# res_scale_bounds = [1.5, 1.55]
# input_scale_bounds = [0.65, 0.7]
# res_scale_bounds = [1.62, 1.92]
# input_scale_bounds = [1, 1.3]
# res_scale_bounds = [1.73, 1.83]
# input_scale_bounds = [1.1, 1.2]
# res_scale_bounds = [1.777, 1.802]
# input_scale_bounds = [1.137, 1.162]
# res_scale_bounds = [1.785, 1.795]
# input_scale_bounds = [1.145, 1.155]

# res_scale_bounds = [1.70, 1.75]
# input_scale_bounds = [0.25, 0.30]

# Bounds for n_res = 30
# res_scale_bounds = [0, 2]
# input_scale_bounds = [0, 2]
# res_scale_bounds = [1.75, 2.05]
# input_scale_bounds = [1, 1.3]
# res_scale_bounds = [1.87, 1.97]
# input_scale_bounds = [1.1, 1.2]
# get current date
now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
save_name = f"{now}_{mode}{kernel_size if mode=='rand_conv' else ''}{n_layers if mode=='struct_rand' else ''}x{n_linops}{additional}_seed{seed}_weight{weight_scale_bounds}_bias{bias_scale_bounds}"

metric_erf = stability_test(
    width=width,
    depth=depth,
    mode=mode,

    n_channels = n_channels,
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
    res_scale_bounds=weight_scale_bounds,
    input_scale_bounds=bias_scale_bounds,

    device=device,
    seed=seed,
)

plt.figure()
seaborn.set_style("whitegrid")
img = metric_erf.T
threshold = 1e-5
img[img < threshold] = threshold
input_min = 0
input_max = 1
res_min = 0
res_max = 1
plt.imshow(
    img[
        int(input_min * resolution) : int(input_max * resolution),
        int(res_min * resolution) : int(res_max * resolution),
    ],
    norm=matplotlib.colors.LogNorm(vmin=1e-10, vmax=1),
)  #

ax = plt.gca()
plt.grid(False)
plt.clim(threshold, 1)
plt.colorbar()

input_scale_min = bias_scale_bounds[0] + input_min * (
    bias_scale_bounds[1] - bias_scale_bounds[0]
)
input_scale_max = bias_scale_bounds[0] + input_max * (
    bias_scale_bounds[1] - bias_scale_bounds[0]
)
res_scale_min = weight_scale_bounds[0] + res_min * (
    weight_scale_bounds[1] - weight_scale_bounds[0]
)
res_scale_max = weight_scale_bounds[0] + res_max * (
    weight_scale_bounds[1] - weight_scale_bounds[0]
)
ylab = np.linspace(input_scale_min, input_scale_max, num=int(bias_scale_bounds[1] + 1))
xlab = np.linspace(res_scale_min, res_scale_max, num=int(weight_scale_bounds[1] + 1))
indXx = np.linspace(0, resolution - 1, num=xlab.shape[0]).astype(int)
indXy = np.linspace(0, resolution - 1, num=ylab.shape[0]).astype(int)

ax.set_xticks(indXx)
ax.set_xticklabels(xlab)
ax.set_yticks(indXy)
ax.set_yticklabels(ylab)
ax.set_xlabel("Weight scale")
ax.set_ylabel("Bias scale")
ax.set_title("Asymptotic stability metric\nfor $f=$erf")

if save is True:
    if not os.path.exists("data/" + save_name):
        os.makedirs("data/" + save_name)
    np.save("data/" + save_name + "/" + "metric_erf.npy", metric_erf)
    np.save("data/" + save_name + "/" + "xlab.npy", xlab)
    np.save("data/" + save_name + "/" + "ylab.npy", ylab)
    plt.savefig("fig/" + save_name + ".png")

plt.show()
