import os
from warnings import warn

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn
import torch


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


def plot_frontier(
    errs,
    W_scale_range,
    b_scale_range,
    resolution,
    save_path=None,
):
    plt.figure()
    seaborn.set_style("whitegrid")
    img = errs.T
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

    bias_scale_min = b_scale_range[0] + bias_min * (b_scale_range[1] - b_scale_range[0])
    bias_scale_max = b_scale_range[0] + bias_max * (b_scale_range[1] - b_scale_range[0])
    weight_scale_min = W_scale_range[0] + weight_min * (
        W_scale_range[1] - W_scale_range[0]
    )
    weight_scale_max = W_scale_range[0] + weight_max * (
        W_scale_range[1] - W_scale_range[0]
    )
    ylab = np.linspace(bias_scale_min, bias_scale_max, num=int(b_scale_range[1] + 1))
    xlab = np.linspace(
        weight_scale_min, weight_scale_max, num=int(W_scale_range[1] + 1)
    )
    indXx = np.linspace(0, resolution - 1, num=xlab.shape[0]).astype(int)
    indXy = np.linspace(0, resolution - 1, num=ylab.shape[0]).astype(int)

    ax.set_xticks(indXx)
    ax.set_xticklabels(xlab)
    ax.set_yticks(indXy)
    ax.set_yticklabels(ylab)
    ax.set_xlabel("Weight variance")
    ax.set_ylabel("Bias variance")

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    plt.show()

    return
