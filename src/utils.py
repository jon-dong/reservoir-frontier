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
    b_min = 0
    b_max = 1
    W_min = 0
    W_max = 1
    plt.imshow(
        img[
            int(b_min * resolution[1]) : int(b_max * resolution[1]),
            int(W_min * resolution[0]) : int(W_max * resolution[0]),
        ],
        norm=matplotlib.colors.LogNorm(vmin=1e-10, vmax=1),
    )  #

    ax = plt.gca()
    plt.grid(False)
    plt.clim(threshold, 1)
    plt.colorbar()

    b_scale_min = b_scale_range[0] + b_min * (b_scale_range[1] - b_scale_range[0])
    b_scale_max = b_scale_range[0] + b_max * (b_scale_range[1] - b_scale_range[0])
    W_scale_min = W_scale_range[0] + W_min * (W_scale_range[1] - W_scale_range[0])
    W_scale_max = W_scale_range[0] + W_max * (W_scale_range[1] - W_scale_range[0])
    ylab = np.linspace(b_scale_min, b_scale_max, num=int(b_scale_range[1] + 1))
    xlab = np.linspace(W_scale_min, W_scale_max, num=int(W_scale_range[1] + 1))
    indXx = np.linspace(0, resolution[0] - 1, num=xlab.shape[0]).astype(int)
    indXy = np.linspace(0, resolution[1] - 1, num=ylab.shape[0]).astype(int)

    ax.set_xticks(indXx)
    ax.set_xticklabels(xlab)
    ax.set_yticks(indXy)
    ax.set_yticklabels(ylab)
    ax.set_xlabel("Weight variance")
    ax.set_ylabel("Bias variance")

    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        np.save(save_path + "metric_erf.npy", errs)
        np.save(save_path + "xlab.npy", xlab)
        np.save(save_path + "ylab.npy", ylab)
        plt.savefig(save_path + "frontier.pdf")
    plt.show()

    return


def linear_regression(X, Y):
    """
    Computes least squares linear regression: Y = H*X + V

    Args:
        X: 1D numpy array (independent variable)
        Y: 1D numpy array (dependent variable)

    Returns:
        Tuple of (H, V) where H is slope and V is intercept
    """
    if X.ndim != 1 or Y.ndim != 1:
        raise ValueError("Both X and Y must be 1D arrays.")
    if X.size != Y.size:
        raise ValueError("X and Y must be the same length.")

    # Construct design matrix
    A = np.vstack([X, np.ones_like(X)]).T

    # Solve least squares
    H, V = np.linalg.lstsq(A, Y, rcond=None)[0]

    return H, V
