# %%
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn
import torch

from src.fractal import generate_input
from src.network import Network
from src.utils import get_freer_gpu, plot_frontier

# %%
dtype = torch.float32
device = get_freer_gpu()
device

# %%
W_scale_range = [0, 4]
b_scale_range = [0, 4]
# W_scale_range = [2.0, 2.4]
# b_scale_range = [1.8, 2.2]
# W_scale_range = [2.08, 2.12]
# b_scale_range = [1.98, 2.02]
# W_scale_range = [2.098, 2.102]
# b_scale_range = [1.998, 2.002]
# W_scale_range = [2.1875, 2.2125]
# b_scale_range = [2.0375, 2.0625]
# W_scale_range = [2.1625, 2.1875]
# b_scale_range = [2.0375, 2.0625]
# W_scale_range = [2.168, 2.193]
# b_scale_range = [2.0375, 2.0625]
torch.manual_seed(3)
resolution = 1000
W_scales = torch.linspace(W_scale_range[0], W_scale_range[1], steps=resolution).to(
    device
)
b_scales = torch.linspace(b_scale_range[0], b_scale_range[1], steps=resolution).to(
    device
)
n_chunks = 1
b_scales_chunks = b_scales.chunk(n_chunks)

# %%
width = 20
depth = 1000
network = Network(
    width=width,
    depth=depth,
    mode="conv",
    config={"kernel_size": width, "n_layers": 1.5, "mags": ["unit"], "osr": 1.01},
    device=device,
)

# %%
bs = torch.randn(depth, width).to(device)

input1, input2 = generate_input(width, mode="independent", device=device, dtype=dtype)

outputs1 = []
outputs2 = []
for b_scale_chunk in b_scales_chunks:
    outputs1.append(
        network.forward(input1, bs=bs, W_scales=W_scales, b_scales=b_scale_chunk)[0]
    )
    outputs2.append(
        network.forward(input2, bs=bs, W_scales=W_scales, b_scales=b_scale_chunk)[0]
    )

outputs1 = torch.cat(outputs1, dim=1)
outputs2 = torch.cat(outputs2, dim=1)

# %%
outputs1.squeeze_()
outputs2.squeeze_()
errs = torch.sum((outputs1 - outputs2) ** 2, dim=2).cpu().numpy()

plot_frontier(
    errs,
    W_scale_range,
    b_scale_range,
    resolution,
    # save_path="data/frontier_plot.png",
)
