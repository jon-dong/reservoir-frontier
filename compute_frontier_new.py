import datetime
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn
import torch
from tqdm import tqdm

from reservoir import CustomReservoir
from utils import stability_test

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # TOCHANGE 1/2

seed = 0
res_size = 100
input_size = 100
input_len = 10000
resolution = 1000
# Bounds for n_res = 100
# res_scale_bounds = [0, 2]
# input_scale_bounds = [0, 2]
res_scale_bounds = [1.62, 1.92]
input_scale_bounds = [1, 1.3]
# res_scale_bounds = [1.73, 1.83]
# input_scale_bounds = [1.1, 1.2]
# res_scale_bounds = [1.777, 1.802]
# input_scale_bounds = [1.137, 1.162]
# res_scale_bounds = [1.785, 1.795]
# input_scale_bounds = [1.145, 1.155]

# Bounds for n_res = 30
# res_scale_bounds = [0, 2]
# input_scale_bounds = [0, 2]
# res_scale_bounds = [1.75, 2.05]
# input_scale_bounds = [1, 1.3]
# res_scale_bounds = [1.87, 1.97]
# input_scale_bounds = [1.1, 1.2]
# get current date
now = datetime.datetime.now()
filename = f"{now}res{res_scale_bounds}_input{input_scale_bounds}_HR"  # TOCHANGE 2/2

metric_erf = stability_test(
    res_size=res_size,
    input_size=input_size,
    input_len=input_len,
    resolution=resolution,
    constant_input=False,
    res_scale_bounds=res_scale_bounds,
    input_scale_bounds=input_scale_bounds,
    device=device,
    seed=seed,
    use='network',
)

plt.figure()
seaborn.set_style("whitegrid")
img = metric_erf.T
threshold = 1e-5
img[img < threshold] = threshold
input_min = 0
input_max = 1
res_min = 0
res_max = 1
plt.imshow(
    img[
        int(input_min * resolution) : int(input_max * resolution),
        int(res_min * resolution) : int(res_max * resolution),
    ],
    norm=matplotlib.colors.LogNorm(vmin=1e-10, vmax=1),
)  #

ax = plt.gca()
plt.grid(False)
plt.clim(threshold, 1)
plt.colorbar()

input_scale_min = input_scale_bounds[0] + input_min * (
    input_scale_bounds[1] - input_scale_bounds[0]
)
input_scale_max = input_scale_bounds[0] + input_max * (
    input_scale_bounds[1] - input_scale_bounds[0]
)
res_scale_min = res_scale_bounds[0] + res_min * (
    res_scale_bounds[1] - res_scale_bounds[0]
)
res_scale_max = res_scale_bounds[0] + res_max * (
    res_scale_bounds[1] - res_scale_bounds[0]
)
ylab = np.linspace(input_scale_min, input_scale_max, num=int(input_scale_bounds[1] + 1))
xlab = np.linspace(res_scale_min, res_scale_max, num=int(res_scale_bounds[1] + 1))
indXx = np.linspace(0, resolution - 1, num=xlab.shape[0]).astype(int)
indXy = np.linspace(0, resolution - 1, num=ylab.shape[0]).astype(int)

ax.set_xticks(indXx)
ax.set_xticklabels(xlab)
ax.set_yticks(indXy)
ax.set_yticklabels(ylab)
ax.set_xlabel("Reservoir scale")
ax.set_ylabel("Input scale")
ax.set_title("Asymptotic stability metric\nfor $f=$erf")

np.save("data/" + filename + ".npy", metric_erf)
np.save("data/" + filename + "_xlab.npy", xlab)
np.save("data/" + filename + "_ylab.npy", ylab)
plt.savefig("fig/" + filename + ".png")

plt.show()
