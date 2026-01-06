import numpy as np
import torch
from tqdm import trange

from .network import Network


def generate_input(
    width,
    mode,
    noise_level=None,  # 0.01
    x1=None,
    x2=None,
    dtype=torch.float32,
    device="cpu",
):
    """
    Generates pairs of input vectors for information propagation testing.

    Args:
        width: Dimension of input vectors
        mode: Either "independent" (two random vectors) or "sensitivity" (perturbed pair)
        noise_level: Perturbation level for sensitivity mode
        x1, x2: Optional pre-generated inputs
        dtype: PyTorch data type
        device: PyTorch device

    Returns:
        Tuple of (x1, x2) normalized input vectors
    """
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


def propagation_test_net(
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
    Tests information propagation through network with different inputs.

    Computes distance between network outputs for two different inputs across
    different reservoir scales.

    Args:
        net: Network instance
        x1, x2: Input vectors (generated if None)
        bs: Bias vectors
        W_scales: Weight scale values to test
        b_scales: Bias scale values to test
        mode: Input generation mode ("independent" or "sensitivity")
        noise_level: Perturbation level for sensitivity mode
        normalize: Whether to normalize network states
        n_save_last: Number of final timesteps to save

    Returns:
        Distance tensor of shape (n_scales, n_history)
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

    outputs2 = net.forward(
        x2,
        bs,
        W_scales=W_scales,
        b_scales=b_scales,
        normalize=normalize,
        n_save_last=n_save_last,
    )

    return torch.sum((outputs1 - outputs2) ** 2, dim=-1)


def propagation_test(
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
    propagation_mode=None,
    noise_level=None,
    n_repeats=1,
    n_save_last=1,
    chunks=[1, 1],
    device="cpu",
    dtype=torch.float32,
    seed=0,
):
    """
    Tests information propagation across parameter grid.

    Creates a grid of (W_scale, b_scale) values and tests how information
    propagates through the network at each point.

    Args:
        width: Network width (state size)
        depth: Network depth (number of layers)
        mode: Network architecture mode ('rand', 'struct', 'conv')
        W_scale_bounds: [min, max] weight scale range
        b_scale_bounds: [min, max] bias scale range
        resolution: [n_W_scales, n_b_scales] grid resolution
        config_linop: Linear operator configuration
        config_resid: Residual configuration
        constant_bias: Whether to use constant bias across layers
        normalize: Whether to normalize network states
        propagation_mode: Input generation mode for propagation test
        noise_level: Perturbation level for sensitivity mode
        n_repeats: Number of repetitions to average
        n_save_last: Number of final timesteps to save
        chunks: [n_W_chunks, n_b_chunks] for chunked computation
        device: PyTorch device
        dtype: PyTorch data type
        seed: Random seed

    Returns:
        Error tensor of shape (n_save_last, n_W_scales, n_b_scales)
    """
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
                errs[:, W_start:W_end, b_start:b_end] += propagation_test_net(
                    models[i],
                    x1=input1,
                    x2=input2,
                    bs=bs,
                    W_scales=W_scales,
                    b_scales=b_scales,
                    mode=propagation_mode,
                    noise_level=noise_level,
                    normalize=normalize,
                    n_save_last=n_save_last,
                )  # return size (n_save_last, *resolution)
    errs = errs / n_repeats  # normalize

    return errs


def gradient_divergence_test_net(
    net,
    x=None,
    bs=None,
    W_scale=1.0,
    b_scale=1.0,
    normalize=False,
    n_save_last=1,
):
    """
    Tests gradient divergence through backpropagation with two random loss gradients.

    Performs a forward pass, then backpropagates two independent random gradient
    signals from the output layer and measures their L2 distance at each layer.

    Args:
        net: Network instance
        x: Input vector (generated if None)
        bs: Bias vectors
        W_scale: Weight scale value
        b_scale: Bias scale value
        normalize: Whether to normalize network states
        n_save_last: Number of final layers to save gradient divergence for

    Returns:
        Gradient divergence tensor of shape (n_save_last,)
    """
    bs = bs.to(net.device, net.dtype)

    if x is None:
        x = torch.randn(net.width).to(net.device, net.dtype)
        x = x / torch.norm(x)

    # Ensure gradient tracking is enabled
    with torch.enable_grad():
        # Store layer activations during forward pass
        layer_activations = []
        # Create input tensor that requires grad
        x_current = x.clone().detach().requires_grad_(True)
        layer_activations.append(x_current)

        # Transform biases
        bs_transformed = torch.einsum("ij,nj -> ni", net.W_bias, bs)

        # Forward pass through all layers, storing activations
        counter = 0
        for i in range(net.depth):
            # Apply linear operator with gradient tracking enabled
            if x_current.ndim == 1:
                x_expanded = x_current[None, None, :]
            else:
                x_expanded = x_current

            Wx = net.linops[counter].apply(x_expanded, track_grad=True)
            if Wx.ndim == 3:
                Wx = Wx.squeeze(0).squeeze(0)

            # Apply scaling and bias
            z = W_scale * Wx + b_scale * bs_transformed[i, :]

            # Apply activation
            if torch.is_complex(z):
                x_new = net.activation(z.real) * np.sqrt(2)
            else:
                x_new = net.activation(z)

            if net.mode == "rand":
                x_new = x_new / np.sqrt(net.width)

            # Handle residual connections
            if net.resid_span is not None and i >= net.resid_span:
                if i > 0 and i % net.resid_stride == 0:
                    x_new = x_new + layer_activations[i - net.resid_span + 1]

            # Normalize if needed
            if normalize:
                x_new = (x_new - torch.mean(x_new)) / (torch.std(x_new) + 1e-10)

            # Don't manually set requires_grad - PyTorch tracks it automatically
            layer_activations.append(x_new)
            x_current = x_new

            counter += 1
            if counter == net.n_linops:
                counter = 0

        # Output is the last activation
        output = layer_activations[-1]

        # Create two random loss gradients (normalized)
        grad1 = torch.randn_like(output)
        grad1 = grad1 / torch.norm(grad1)
        grad2 = torch.randn_like(output)
        grad2 = grad2 / torch.norm(grad2)

        # Compute which layers to track
        if n_save_last > net.depth:
            n_save_last = net.depth + 1  # Include input layer

        layers_to_track = list(range(net.depth + 1 - n_save_last, net.depth + 1))
        activations_to_track = [layer_activations[i] for i in layers_to_track]

        # Backpropagate both gradients in one pass for all tracked layers
        grads1_all = torch.autograd.grad(
            outputs=output,
            inputs=activations_to_track,
            grad_outputs=grad1,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )

        grads2_all = torch.autograd.grad(
            outputs=output,
            inputs=activations_to_track,
            grad_outputs=grad2,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )

        # Compute L2 distances for all tracked layers
        gradient_distances = []
        for grads1, grads2 in zip(grads1_all, grads2_all):
            if grads1 is None or grads2 is None:
                # If gradient is None, that activation wasn't used in the computation
                gradient_distances.append(torch.tensor(0.0, device=net.device))
            else:
                distance = torch.norm(grads1 - grads2)
                gradient_distances.append(distance)

        return torch.stack(gradient_distances)


def gradient_divergence_test(
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
    n_repeats=1,
    n_save_last=1,
    chunks=[1, 1],
    device="cpu",
    dtype=torch.float32,
    seed=0,
):
    """
    Tests gradient divergence across parameter grid.

    Creates a grid of (W_scale, b_scale) values and tests how two random
    gradient signals diverge as they backpropagate through the network.

    Args:
        width: Network width (state size)
        depth: Network depth (number of layers)
        mode: Network architecture mode ('rand', 'struct', 'conv')
        W_scale_bounds: [min, max] weight scale range
        b_scale_bounds: [min, max] bias scale range
        resolution: [n_W_scales, n_b_scales] grid resolution
        config_linop: Linear operator configuration
        config_resid: Residual configuration
        constant_bias: Whether to use constant bias across layers
        normalize: Whether to normalize network states
        n_repeats: Number of repetitions to average
        n_save_last: Number of final layers to save gradient divergence for
        chunks: [n_W_chunks, n_b_chunks] for chunked computation
        device: PyTorch device
        dtype: PyTorch data type
        seed: Random seed

    Returns:
        Gradient divergence tensor of shape (n_save_last, n_W_scales, n_b_scales)
    """
    torch.manual_seed(seed)

    # Generate biases
    if not constant_bias:
        bs = torch.randn(depth, width).to(device)
        for i in range(depth):
            bs[i, :] = bs[i, :] / torch.norm(bs[i, :])
    else:
        bs = torch.randn(width).to(device)
        bs = bs / torch.norm(bs)
        bs = bs.repeat(depth, 1)

    # Create scale grids
    n_W_scales = resolution[0]
    n_b_scales = resolution[1]
    W_scales = np.linspace(W_scale_bounds[0], W_scale_bounds[1], num=n_W_scales)
    b_scales = np.linspace(b_scale_bounds[0], b_scale_bounds[1], num=n_b_scales)

    # Initialize W_bias and input
    W_bias = torch.randn(width, width).to(device)
    input_x = torch.randn(width).to(device)
    input_x = input_x / torch.norm(input_x)

    # Create models for n_repeats
    models = []
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

    # Initialize results tensor
    grad_divs = torch.zeros(n_save_last, n_W_scales, n_b_scales).to(device)

    # Iterate over repeats
    for repeat_idx in range(n_repeats):
        print(f"Repeat {repeat_idx + 1}/{n_repeats}")
        # Iterate over parameter grid
        for W_idx in trange(n_W_scales, desc="W_scales"):
            for b_idx in range(n_b_scales):
                W_scale = W_scales[W_idx]
                b_scale = b_scales[b_idx]

                # Compute gradient divergence for this parameter combination
                grad_div = gradient_divergence_test_net(
                    models[repeat_idx],
                    x=input_x,
                    bs=bs,
                    W_scale=W_scale,
                    b_scale=b_scale,
                    normalize=normalize,
                    n_save_last=n_save_last,
                )

                grad_divs[:, W_idx, b_idx] += grad_div

    # Average over repeats
    grad_divs = grad_divs / n_repeats

    return grad_divs


# Backward compatibility aliases (old names from stability testing)
stability_test = propagation_test
stability_test_net = propagation_test_net
