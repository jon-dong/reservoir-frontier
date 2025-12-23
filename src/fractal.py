import os

from matplotlib import pyplot as plt
import numpy as np
import porespy as ps
import torch
from tqdm import tqdm

from .network import Network


def erf_frontier(res_scale):
    return np.sqrt(
        4 * res_scale**4 / np.pi**2
        - 1 / (4)
        - 2
        * res_scale**2
        / np.pi
        * np.arcsin((16 * res_scale**4 - np.pi**2) / (16 * res_scale**4 + np.pi**2))
        + 1e-6
    )


def extract_edges(X):
    """
    Detects edges in a signed 2D array by identifying sign changes.

    For each 2x2 cell in the array, an edge is detected if the four corner
    values contain both positive and negative signs.

    Args:
        X: 2D numpy array with signed values

    Returns:
        Binary numpy array of shape (H-1, W-1) where True indicates an edge
    """

    Y = np.stack((X[1:, 1:], X[:-1, 1:], X[1:, :-1], X[:-1, :-1]), axis=-1)
    Z = np.max(Y, axis=-1) * np.min(Y, axis=-1)
    return Z < 0


def estimate_fractal_dimension(hist_video, title_plot=None):
    """
    Estimates the fractal dimension of a sequence of images

    hist_video: a list of images as numpy arrays containing the convergence/divergence scheme
    return: the median fractal dimension estimate
    """
    edges = [extract_edges(U) for U in hist_video]  # U[0]
    box_counts = [ps.metrics.boxcount(U) for U in edges]
    all_images = np.concatenate([bc.slope for bc in box_counts])

    if title_plot is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
        fig.suptitle(title_plot)
        ax1.set_yscale("log")
        ax1.set_xscale("log")
        ax1.set_xlabel("box edge length")
        ax1.set_ylabel("number of boxes spanning phases")
        ax2.set_xlabel("box edge length")
        ax2.set_ylabel("Fractal Dimension")
        ax2.set_xscale("log")

        for bc in box_counts:  # plotting for each zooming
            print(f"Box size: {bc.size}, fractal dimension estimate: {bc.slope}")
            ax1.plot(bc.size, bc.count, "-o")
            ax2.plot(bc.size, bc.slope, "-o")
        plt.savefig(title_plot + ".pdf")

    mfd = np.median(
        all_images
    )  # getting the median of slopes (10 box sizes) over all the images
    print(f"median fractal dimension estimate {mfd}")

    return mfd


def linear_regression(X, Y):
    ##    Computes least squares linear regression: Y = H*X + V

    if X.ndim != 1 or Y.ndim != 1:
        raise ValueError("Both X and Y must be 1D arrays.")
    if X.size != Y.size:
        raise ValueError("X and Y must be the same length.")

    # Construct design matrix
    A = np.vstack([X, np.ones_like(X)]).T

    # Solve least squares
    H, V = np.linalg.lstsq(A, Y, rcond=None)[0]

    return H, V


def count_boxes(field: np.ndarray, box_size: int):
    """
    Counts boxes of a given size that contain at least one True value.

    Used in box-counting method for fractal dimension estimation. Divides the
    2D boolean array into a grid of box_size×box_size boxes and counts how many
    boxes contain at least one True value.

    Args:
        field: 2D boolean numpy array
        box_size: Size of each box (must be >= 1)

    Returns:
        Tuple of (N_boxes, box_size) where N_boxes is the count of occupied boxes
    """
    if box_size < 1:
        raise ValueError("box_size must be at least 1")
    H, W = field.shape

    # Pad array to make dimensions divisible by box_size
    pad_H = (box_size - H % box_size) % box_size
    pad_W = (box_size - W % box_size) % box_size

    if pad_H > 0 or pad_W > 0:
        padded = np.pad(field, ((0, pad_H), (0, pad_W)), mode='constant', constant_values=False)
    else:
        padded = field

    H_padded, W_padded = padded.shape
    n_rows = H_padded // box_size
    n_cols = W_padded // box_size

    # Reshape into blocks: (n_rows, box_size, n_cols, box_size)
    reshaped = padded.reshape(n_rows, box_size, n_cols, box_size)

    # Check if any element in each block is True (reduce over box dimensions)
    occupied = np.any(reshaped, axis=(1, 3))

    # Count occupied boxes
    N_boxes = np.sum(occupied)

    return N_boxes, box_size


def compute_dim(X, min_idx, max_idx, scales=list(range(1, 50))):
    """
    Computes fractal dimension using the box-counting method.

    Counts occupied boxes at multiple scales, then estimates the fractal
    dimension as the negative slope of the log-log plot of box count vs box size.
    Linear regression is performed on a subset of scales specified by min_idx and max_idx.

    Args:
        X: 2D boolean numpy array (typically output from extract_edges)
        min_idx: Start index for linear regression range
        max_idx: End index for linear regression range
        scales: List of box sizes to test (default: range(1, 50))

    Returns:
        Tuple of (fractal_dim, intercept, log_count, log_scales) where:
        - fractal_dim: Estimated fractal dimension (-slope)
        - intercept: Y-intercept of the regression line
        - log_count: Log of box counts at all scales
        - log_scales: Log of all box sizes
    """
    count_through_scales = []
    int_scales = []
    for scale in scales:
        count, int_scale = count_boxes(X, scale)
        count_through_scales.append(count)
        int_scales.append(int_scale)

    log_count = np.log(np.array(count_through_scales))
    log_scales = np.log(np.array(int_scales))
    H, V = linear_regression(log_scales[min_idx:max_idx], log_count[min_idx:max_idx])

    return -H, V, log_count, log_scales


def fractal_dim_folder(folder, title_plot=None):
    """
    Computes the fractal dimension for all the .npz files contained in a specific folder
    """
    hist_video = []
    for file in os.listdir(folder):
        img = np.load(folder + file, allow_pickle=True)
        img[img < 1e-3] = -1
        img[img >= 1e-3] = 1
        hist_video.append(img)

    estimate_fractal_dimension(hist_video, title_plot)


def generate_input(
    width,
    mode,
    noise_level=None,  # 0.01
    x1=None,
    x2=None,
    dtype=torch.float32,
    device="cpu",
):
    if mode == "independent":
        if x1 is None and x2 is None:
            x1 = torch.randn(width).to(device, dtype)
            x1 = x1 / torch.norm(x1)
            x2 = torch.randn(width).to(device, dtype)
            x2 = x2 / torch.norm(x2)
    elif mode == "sensitivity":
        x1 = torch.randn(width).to(device, dtype)
        epsilon = noise_level * torch.randn(width).to(device, dtype)
        x2 = x1 + epsilon

        x1 = x1 / torch.norm(x1)
        x2 = x2 / torch.norm(x2)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return x1, x2


def stability_test_net(
    net,
    x1=None,
    x2=None,
    bs=None,
    W_scales=None,
    b_scales=None,
    mode="independent",
    noise_level=0.01,
    normalize=False,
    n_save_last=1,
):
    """
    Stability test on the same input and different reservoir scales

    Follows the distance between the reservoir states through time, whether they converge to the same trajectory

    Returns:
    dist: shape (n_scales, n_history)
    """
    bs = bs.to(net.device, net.dtype)

    x1, x2 = generate_input(
        width=net.width,
        mode=mode,
        noise_level=noise_level,
        x1=x1,
        x2=x2,
        dtype=net.dtype,
        device=net.device,
    )

    outputs1 = net.forward(
        x1,
        bs,
        W_scales=W_scales,
        b_scales=b_scales,
        normalize=normalize,
        n_save_last=n_save_last,
    )

    net.counter = 0
    outputs2 = net.forward(
        x2,
        bs,
        W_scales=W_scales,
        b_scales=b_scales,
        normalize=normalize,
        n_save_last=n_save_last,
    )

    return torch.sum((outputs1 - outputs2) ** 2, dim=-1)


def stability_test(
    width,
    depth,
    mode,
    W_scale_bounds=[0, 4],
    b_scale_bounds=[0, 4],
    resolution=None,
    config_linop=None,
    config_resid=None,
    constant_bias=False,
    normalize=False,
    stability_mode=None,
    noise_level=None,
    n_repeats=1,
    n_save_last=1,
    # average=1,
    chunks=[1, 1],
    device="cpu",
    dtype=torch.float32,
    seed=0,
):
    torch.manual_seed(seed)
    if not constant_bias:
        bs = torch.randn(depth, width).to(device)
        for i in range(depth):  # normalize at each time step
            bs[i, :] = bs[i, :] / torch.norm(bs[i, :])
    else:
        bs = torch.randn(width).to(device)
        bs = bs / torch.norm(bs)
        bs = bs.repeat(depth, 1)

    n_W_scales = resolution[0]
    n_b_scales = resolution[1]
    W_scales = np.linspace(W_scale_bounds[0], W_scale_bounds[1], num=n_W_scales)
    b_scales = np.linspace(b_scale_bounds[0], b_scale_bounds[1], num=n_b_scales)
    n_W_chunks = chunks[0]
    n_b_chunks = chunks[1]
    W_scales_chunked = torch.chunk(torch.tensor(W_scales), n_W_chunks)
    b_scales_chunked = torch.chunk(torch.tensor(b_scales), n_b_chunks)
    # Initialize
    W_bias = torch.randn(width, width).to(device)
    # W_bias = torch.tensor(1.0)
    input1 = torch.randn(width).to(device)
    input1 = input1 / torch.norm(input1)
    input2 = torch.randn(width).to(device)
    input2 = input2 / torch.norm(input2)

    models = []

    # make sure to use the same instance
    for _ in range(n_repeats):
        model = Network(
            width=width,
            depth=depth,
            mode=mode,
            W_bias=W_bias,
            config_linop=config_linop,
            config_resid=config_resid,
            dtype=dtype,
            device=device,
        )
        models.append(model)

    errs = torch.zeros(n_save_last, n_W_scales, n_b_scales).to(device)
    for i in range(n_repeats):
        for W_idx, W_scales in enumerate(W_scales_chunked):
            for b_idx, b_scales in enumerate(b_scales_chunked):
                W_start = W_idx * n_W_scales // n_W_chunks
                W_end = W_start + n_W_scales // n_W_chunks
                b_start = b_idx * n_b_scales // n_b_chunks
                b_end = b_start + n_b_scales // n_b_chunks
                errs[:, W_start:W_end, b_start:b_end] += stability_test_net(
                    models[i],
                    x1=input1,
                    x2=input2,
                    bs=bs,
                    W_scales=W_scales,
                    b_scales=b_scales,
                    mode=stability_mode,
                    noise_level=noise_level,
                    normalize=normalize,
                    n_save_last=n_save_last,
                )  # return size (n_save_last, *resolution)
    errs = errs / n_repeats  # normalize
    # errs = torch.mean(errs[-average:], dim=0)

    return errs
