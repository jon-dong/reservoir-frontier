import datetime
import os
from warnings import warn

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn
import torch

from utils import stability_test1d

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
            return torch.device("cpu")

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
width = 101 # state size
depth = 99 # input length for reservoir
mode = "rand" # in ['rand', 'struct', 'conv']
additional = '' # additional name for saving

normalize = False # layer normalization
n_channels = 1 # multiple networks and average errors
n_linops = depth # number of linops to iterate on
residual_length = None # residual connection length
residual_interval = None # residual connection interval

stability_mode = "independent" # in ['sensitivity', 'independent']
noise_level = 1e-15 # for sensitivity analysis

# struct
n_layers = 1.5
mags = ['unit'] # in ['marchenko', 'unit']
osr = 1.01 # oversampling ratio

# conv
kernel_size = width

if mode == 'rand':
    n_layers = None
    mags = None
    osr = None
    kernel_size = None

save = True

weight_scale = 1
bias_scale = 1

timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
save_folder = f"{timestamp}_{mode}_w{width}d{depth}{'_kernel'+str(kernel_size) if mode=='conv' else ''}{'_layer'+str(n_layers) if mode=='struct' else ''}_{additional}_seed{seed}_weight{weight_scale}_bias{bias_scale}/"

metric_erf = stability_test1d(
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
    n_hist=depth,

    stability_mode=stability_mode,
    noise_level=noise_level,
    weight_scale=weight_scale,
    bias_scale=bias_scale,

    device=device,
    seed=seed,
)

print(metric_erf.shape)


plt.figure()
seaborn.set_style("whitegrid")
plt.semilogy(torch.sum(metric_erf, dim=1))
plt.show()