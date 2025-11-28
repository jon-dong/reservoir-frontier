# %%
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn
import torch

from src.network import Network
from src.utils import get_freer_gpu

# %%
device = get_freer_gpu()
device

# %%
W_scale_range = [0, 4]
b_scale_range = [0, 4]
resolution = 1000
W_scales = torch.linspace(W_scale_range[0], W_scale_range[1], steps=resolution).to(
    device
)
b_scales = torch.linspace(b_scale_range[0], b_scale_range[1], steps=resolution).to(
    device
)

# %%
width = 1000
depth = 1000
network = Network(
    width=width,
    depth=depth,
    bias_scale=1.0,
    W_bias=torch.randn(width, width).to(device),
    mode="rand",
    device=device,
)

# %%
bs = torch.randn(depth, width).to(device)
for i in range(depth):  # normalize input at each time step
    bs[i, :] = bs[i, :] / torch.norm(bs[i, :])

input1 = torch.randn(width).to(device)
input1 /= torch.norm(input1)
input2 = torch.randn(width).to(device)
input2 /= torch.norm(input2)

outputs1 = network.forward(input1, bs=bs, W_scales=W_scales, b_scales=b_scales)
outputs2 = network.forward(input2, bs=bs, W_scales=W_scales, b_scales=b_scales)

# %%
outputs1.squeeze_()
outputs2.squeeze_()
errs = torch.sum((outputs1 - outputs2) ** 2, dim=2).cpu().numpy()
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
weight_scale_min = W_scale_range[0] + weight_min * (W_scale_range[1] - W_scale_range[0])
weight_scale_max = W_scale_range[0] + weight_max * (W_scale_range[1] - W_scale_range[0])
ylab = np.linspace(bias_scale_min, bias_scale_max, num=int(b_scale_range[1] + 1))
xlab = np.linspace(weight_scale_min, weight_scale_max, num=int(W_scale_range[1] + 1))
indXx = np.linspace(0, resolution - 1, num=xlab.shape[0]).astype(int)
indXy = np.linspace(0, resolution - 1, num=ylab.shape[0]).astype(int)

ax.set_xticks(indXx)
ax.set_xticklabels(xlab)
ax.set_yticks(indXy)
ax.set_yticklabels(ylab)
ax.set_xlabel("Weight variance")
ax.set_ylabel("Bias variance")
