import numpy as np
import torch
from tqdm import tqdm

from .network import Network


def generate_input(
    width,
    mode,
    dtype,
    device,
    noise_level=None,  # 0.01
):
    if mode == "independent":
        input1 = torch.randn(width).to(device, dtype)
        input1 = input1 / torch.norm(input1)
        input2 = torch.randn(width).to(device, dtype)
        input2 = input2 / torch.norm(input2)
    elif mode == "sensitivity":
        input1 = torch.randn(width).to(device, dtype)
        epsilon = noise_level * torch.randn(width).to(device, dtype)
        input2 = input1 + epsilon

        input1 = input1 / torch.norm(input1)
        input2 = input2 / torch.norm(input2)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return input1, input2


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
    average=1,
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

    W_scales = np.linspace(W_scale_bounds[0], W_scale_bounds[1], num=resolution)
    b_scales = np.linspace(b_scale_bounds[0], b_scale_bounds[1], num=resolution)
    errs = torch.zeros(resolution, resolution)

    # Initialize
    W_bias = torch.randn(width, width).to(device)
    # W_bias = torch.tensor(1.0)
    # * somehow defining the two inputs before model.stability_test() will yield different transient behavior on the the frontier from defining them inside
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

    rc_metric = torch.zeros(n_save_last, resolution, resolution).to(device)
    for i in range(n_repeats):
        rc_metric += stability_test_net(
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
        )  # return size (n_hist, resolution, resolution)
    rc_metric = rc_metric / n_repeats  # normalize
    errs = torch.mean(rc_metric[-average:], dim=0)

    return errs


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

    if mode == "independent":
        if x1 is None and x2 is None:
            x1 = torch.randn(net.width).to(net.device, net.dtype)
            x1 = x1 / torch.norm(x1)
            x2 = torch.randn(net.width).to(net.device, net.dtype)
            x2 = x2 / torch.norm(x2)
    elif mode == "sensitivity":
        x1 = torch.randn(net.width).to(net.device, net.dtype)
        epsilon = noise_level * torch.randn(net.width).to(net.device, net.dtype)
        x2 = x1 + epsilon

        x1 = x1 / torch.norm(x1)
        x2 = x2 / torch.norm(x2)

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
