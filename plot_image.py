import os
import re
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn

path = "./data/"

pattern = re.compile(r'res\[(.*?)\]_input\[(.*?)\]')

resolution = 1000

for folder in os.listdir(path):
    # skip files
    if not os.path.isdir(path + folder):
        continue

    match = pattern.search(folder)
    if match:
        res_values = match.group(1)
        input_values = match.group(2)
        res_scale_bounds = list(map(float, match.group(1).split(',')))
        input_scale_bounds = list(map(float, match.group(2).split(',')))

    fig_name = folder
    metric_erf = np.load(path + folder + "/metric_erf.npy")
    xlab = np.load(path + folder + "/xlab.npy")
    ylab = np.load(path + folder + "/ylab.npy")

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
    ax.set_xlabel("Weight scale")
    ax.set_ylabel("Bias scale")
    ax.set_title("Asymptotic stability metric\nfor $f=$erf")

    plt.savefig(path + folder + "/frontier.png")

    plt.close()
