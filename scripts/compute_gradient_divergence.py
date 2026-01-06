# %% imports
import datetime
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn
import torch

from src.propagation import gradient_divergence_test
from src.utils import get_freer_gpu, plot_frontier

# %% setup
device = get_freer_gpu()
dtype = torch.float32
data_folder = "data/runs/"

# %% parameters
seed = 1
width = 100  # state size (max 100 for memory)
depth = 100  # number of layers (max 100 for memory)
mode = "struct"  # in ['rand', 'struct', 'conv']
extra = "grad_div"  # additional name for saving
save = True

normalize = False  # layer normalization
n_linops = depth  # number of linops to iterate on
resid_span = None  # residual connection length
resid_stride = None  # residual connection interval

# struct
n_layers = 2
mags = ["unit", "unit"]  # in ['marchenko', 'unit']
osr = 1.01  # oversampling ratio

# conv
kernel_size = 100

config_linop = {
    "n_linops": n_linops,
    "n_layers": n_layers,
    "mags": mags,
    "osr": osr,
    "kernel_size": kernel_size,
}
config_resid = {
    "resid_span": resid_span,
    "resid_stride": resid_stride,
}

if mode == "rand":
    n_layers = None
    mags = None
    osr = None
    kernel_size = None

resolution = [50, 50]  # number of weight and bias scales (reduced for memory)
chunks = [1, 1]
n_save_last = depth  # save all layers to see gradient divergence evolution
n_repeats = 1  # number of repeats to average over

# Parameter bounds
W_scale_bounds = [1.5, 2.5]
b_scale_bounds = [1.5, 2.5]

# get current date
timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
save_folder = f"{timestamp}_{mode}_w{width}d{depth}{'_kernel' + str(kernel_size) if mode == 'conv' else ''}{'_layer' + str(n_layers) if mode == 'struct' else ''}_{extra}_seed{seed}_weight{W_scale_bounds}_bias{b_scale_bounds}/"

# %% compute gradient divergence
grad_divs = gradient_divergence_test(
    width=width,
    depth=depth,
    mode=mode,
    W_scale_bounds=W_scale_bounds,
    b_scale_bounds=b_scale_bounds,
    resolution=resolution,
    config_linop=config_linop,
    config_resid=config_resid,
    constant_bias=False,
    normalize=normalize,
    n_repeats=n_repeats,
    n_save_last=n_save_last,
    chunks=chunks,
    dtype=dtype,
    device=device,
    seed=seed,
)  # returns shape (n_save_last, n_W_scales, n_b_scales)

# %%
print(f"Gradient divergence shape: {grad_divs.shape}")

# %% plot gradient divergence at final layer (input layer) and save
grad_div_input = grad_divs[0]  # First tracked layer (furthest back - closest to input)
plot_frontier(
    grad_div_input.cpu().numpy(),
    W_scale_bounds,
    b_scale_bounds,
    resolution,
    save_path=None if not save else data_folder + save_folder,
)

# %% plot gradient divergence at output layer
grad_div_output = grad_divs[-1]  # Last tracked layer (output layer)
plt.figure(figsize=(10, 8))
plt.imshow(
    grad_div_output.cpu().numpy().T,
    origin="lower",
    aspect="auto",
    extent=[W_scale_bounds[0], W_scale_bounds[1], b_scale_bounds[0], b_scale_bounds[1]],
    cmap="viridis",
)
plt.colorbar(label="Gradient Divergence (L2)")
plt.xlabel("W_scale")
plt.ylabel("b_scale")
plt.title("Gradient Divergence at Output Layer")
if save:
    os.makedirs(data_folder + save_folder, exist_ok=True)
    plt.savefig(data_folder + save_folder + "gradient_divergence_output.png", dpi=300)
plt.show()

# %% plot gradient divergence evolution across layers (for a specific W_scale, b_scale)
# Pick middle point of the grid
W_idx = resolution[0] // 2
b_idx = resolution[1] // 2

plt.figure(figsize=(10, 6))
layer_indices = list(range(depth + 1 - n_save_last, depth + 1))
plt.plot(layer_indices, grad_divs[:, W_idx, b_idx].cpu().numpy(), "-o")
plt.xlabel("Layer Index")
plt.ylabel("Gradient Divergence (L2)")
plt.title(
    f"Gradient Divergence Evolution\n(W_scale={W_scale_bounds[0] + (W_scale_bounds[1]-W_scale_bounds[0])*W_idx/resolution[0]:.2f}, b_scale={b_scale_bounds[0] + (b_scale_bounds[1]-b_scale_bounds[0])*b_idx/resolution[1]:.2f})"
)
plt.grid(True, alpha=0.3)
plt.yscale("log")
if save:
    os.makedirs(data_folder + save_folder, exist_ok=True)
    plt.savefig(
        data_folder + save_folder + "gradient_divergence_evolution.png", dpi=300
    )
plt.show()

# %% save data
if save:
    os.makedirs(data_folder + save_folder, exist_ok=True)
    torch.save(grad_divs, data_folder + save_folder + "gradient_divergence.pt")
    # Save parameters
    params = {
        "width": width,
        "depth": depth,
        "mode": mode,
        "W_scale_bounds": W_scale_bounds,
        "b_scale_bounds": b_scale_bounds,
        "resolution": resolution,
        "n_save_last": n_save_last,
        "n_repeats": n_repeats,
        "seed": seed,
        "normalize": normalize,
        "config_linop": config_linop,
        "config_resid": config_resid,
    }
    import json

    with open(data_folder + save_folder + "params.json", "w") as f:
        json.dump(params, f, indent=2)
