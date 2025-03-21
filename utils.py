import numpy as np
import torch
from tqdm import tqdm
from reservoir import CustomReservoir
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
    res_size=100,
    input_size=100,
    input_len=100,
    resolution=20,
    constant_input=False,
    res_scale_bounds=[0, 3],
    input_scale_bounds=[0, 2],
    n_linops=1,
    n_layers=None,
    mags=None,
    osr=None,
    kernel_size=None,
    average=10,
    device="cpu",
    seed=0,
    use='reservoir',
    mode=None,
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
        sequence = torch.randn(input_len, input_size).to(device)
        for i in range(input_len):  # normalize input at each timestep
            sequence[i, :] = sequence[i, :] / torch.norm(sequence[i, :])
    else:
        sequence = torch.randn(input_size).to(device)
        sequence = sequence / torch.norm(sequence)
        sequence = sequence.repeat(input_len, 1)

    res_scales = np.linspace(
        res_scale_bounds[0], res_scale_bounds[1], num=resolution
    )
    input_scales = np.linspace(
        input_scale_bounds[0], input_scale_bounds[1], num=resolution
    )
    final_metric = torch.zeros(resolution, resolution)

    # Initializel reservoir and initial states
    W_in = torch.randn(res_size, input_size).to(device)
    W_res = torch.randn(res_size, res_size).to(device)
    initial_state1 = torch.randn(res_size).to(device)
    initial_state1 = initial_state1 / torch.norm(initial_state1)
    initial_state2 = torch.randn(res_size).to(device)
    initial_state2 = initial_state2 / torch.norm(initial_state2)

    # make sure to use the same instance
    if use == 'reservoir':
        model = CustomReservoir(
                f="erf",
                input_size=input_size,
                res_size=res_size,
                W_res=W_res,
                W_in=W_in,
                input_scale=None,
                device=device,
            )
    elif use == 'network':
        model = Network(
                input_size=input_size,
                state_size=res_size,
                input_scale=None,
                #! we don't use the presampled W_res
                W_res=None,
                W_in=W_in,
                n_linops=n_linops,
                n_layers=n_layers,
                mags=mags,
                osr=osr,
                kernel_size=kernel_size,
                mode=mode,
                depth=input_len,
                device=device,
            ) 

    for i_in, input_scale in tqdm(enumerate(input_scales)):
        model.input_scale = input_scale
        rc_metric = model.stability_test(
            sequence, res_scales, state1=initial_state1, state2=initial_state2
        )
        final_metric[:, i_in] = torch.mean(rc_metric[:, -average:], dim=1)
    return final_metric
