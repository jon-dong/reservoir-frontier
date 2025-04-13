import numpy as np
import torch
from tqdm import tqdm
from network import Network


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


def stability_test(
    width,
    depth,
    mode,
    resolution=None,
    constant_input=False,
    res_scale_bounds=[0, 3],
    input_scale_bounds=[0, 2],
    n_linops=1,
    n_layers=None,
    n_hist=20,
    mags=None,
    osr=None,
    kernel_size=None,
    n_channels=None,
    residual_length=None,
    residual_interval=None,
    stability_mode=None,
    noise_level=None,
    average=10,
    device="cpu",
    seed=0,
    normalize=False,
):
    """
    Test the stability of the reservoir for different input and reservoir scales.
    :param res_size: number of units in the reservoir
    :param input_size: dimension of the input
    :param input_len: length of the input sequence
    :param resolution: number of points in the grid
    :param constant_input: if True, the input is constant
    :param res_scale_bounds: bounds of the reservoir scale
    :param input_scale_bounds: bounds of the input scale
    :param device: device to run the test
    :return: final_metric: stability metric for each pair of input and reservoir scales
    """
    torch.manual_seed(seed)
    if not constant_input:
        biases = torch.randn(depth, width).to(device)
        for i in range(depth):  # normalize input at each time step
            biases[i, :] = biases[i, :] / torch.norm(biases[i, :])
    else:
        biases = torch.randn(width).to(device)
        biases = biases / torch.norm(biases)
        biases = biases.repeat(depth, 1)

    weight_scales = np.linspace(
        res_scale_bounds[0], res_scale_bounds[1], num=resolution
    )
    bias_scales = np.linspace(
        input_scale_bounds[0], input_scale_bounds[1], num=resolution
    )
    final_metric = torch.zeros(resolution, resolution)

    # Initialize
    W_bias = torch.randn(width, width).to(device)
    #! somehow defining the two inputs before stability_test() will yield different transient behavior on the the frontier from defining them inside
    #! identical code, but different behavior, very confusing
    input1 = torch.randn(width).to(device)
    input1 = input1 / torch.norm(input1)
    input2 = torch.randn(width).to(device)
    input2 = input2 / torch.norm(input2)

    models = []

    # make sure to use the same instance
    for _ in range(n_channels):
        model = Network(
                width=width,
                depth=depth,
                bias_scale=None,
                W_bias=W_bias,
                n_linops=n_linops,
                n_layers=n_layers,
                n_hist=n_hist,
                mags=mags,
                osr=osr,
                kernel_size=kernel_size,
                residual_length=residual_length,
                residual_interval=residual_interval,
                mode=mode,
                device=device,
            ) 
        models.append(model)

    for i_bias, bias_scale in tqdm(enumerate(bias_scales)):
        rc_metric = torch.zeros(resolution, 20).to(device)
        for i in range(n_channels):
            models[i].bias_scale = bias_scale
            rc_metric += models[i].stability_test(
                input1=input1, input2=input2, biases=biases, weight_scales=weight_scales, mode=stability_mode, noise_level=noise_level, normalize=normalize
            ) # return size (resolution, n_hist)
        rc_metric = rc_metric / n_channels # normalize to have same error scale
        final_metric[:, i_bias] = torch.mean(rc_metric[:, -average:], dim=1)
    return final_metric
