import os
from matplotlib import pyplot as plt
import numpy as np
import porespy as ps
import torch
import matplotlib.pyplot as plt
import porespy as ps
import os
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
    weight_scale_bounds=[0, 3],
    bias_scale_bounds=[0, 2],
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
        weight_scale_bounds[0], weight_scale_bounds[1], num=resolution
    )
    bias_scales = np.linspace(
        bias_scale_bounds[0], bias_scale_bounds[1], num=resolution
    )
    final_metric = torch.zeros(resolution, resolution)

    # Initialize
    W_bias = torch.randn(width, width).to(device)
    #! somehow defining the two inputs before model.stability_test() will yield different transient behavior on the the frontier from defining them inside
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


def stability_test1d(
    width,
    depth,
    mode,
    constant_input=False,
    weight_scale=1,
    bias_scale=1,
    n_linops=1,
    n_layers=None,
    n_hist=50,
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

    # Initialize
    W_bias = torch.randn(width, width).to(device)
    #! somehow defining the two inputs before model.stability_test() will yield different transient behavior on the the frontier from defining them inside
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

    rc_metric = torch.zeros(n_hist, width).to(device)
    for i in range(n_channels):
        models[i].bias_scale = bias_scale
        # models[i].weight_scale = weight_scale
        rc_metric += models[i].stability_test1d(
            input1=input1, input2=input2, biases=biases, weight_scale=weight_scale, mode=stability_mode, noise_level=noise_level, normalize=normalize
        ) # return size (resolution, n_hist)
    rc_metric = rc_metric / n_channels # normalize to have same error scale
    return rc_metric
    # final_metric[:, i_bias] = torch.mean(rc_metric[:, -average:], dim=1)
    # return final_metric



def extract_edges(X):
  """
  define edges as sign changes in the scalar representing convergence or
  divergence rate -- on one side of the edge training converges,
  while on the other side of the edge training diverges
  
  X: a numpy array representing the convergence/divergence with signs
  return: a binary numpy array representing the edges
  """

  Y = np.stack((X[1:,1:], X[:-1,1:], X[1:,:-1], X[:-1,:-1]), axis=-1)
  Z = np.sign(np.max(Y, axis=-1)*np.min(Y, axis=-1))
  return Z<0

def estimate_fractal_dimension(hist_video, title_plot=None):
  '''
  Estimates the fractal dimension of a sequence of images

  hist_video: a list of images as numpy arrays containing the convergence/divergence scheme
  return: the median fractal dimension estimate
  '''
  edges = [extract_edges(U) for U in hist_video]#U[0]
  box_counts = [ps.metrics.boxcount(U) for U in edges]
  all_images = np.concatenate([bc.slope for bc in box_counts])


  if title_plot is not None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    fig.suptitle(title_plot)
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    ax1.set_xlabel('box edge length')
    ax1.set_ylabel('number of boxes spanning phases')
    ax2.set_xlabel('box edge length')
    ax2.set_ylabel('Fractal Dimension')
    ax2.set_xscale('log')

    for bc in box_counts: # plotting for each zooming
      print(f'Box size: {bc.size}, fractal dimension estimate: {bc.slope}')
      ax1.plot(bc.size, bc.count,'-o')
      ax2.plot(bc.size, bc.slope,'-o');
    plt.savefig(title_plot+'.pdf')

  mfd = np.median(all_images) # getting the median of slopes (10 box sizes) over all the images
  print(f'median fractal dimension estimate {mfd}')

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
    #Count how many box_size×box_size (or smaller at the edges) boxes contain at least one True in a 2D boolean array.

    if box_size < 1:
        raise ValueError("box_size must be at least 1")
    H, W = field.shape

    # how many boxes fit (round up to cover the whole image)
    n_rows = int(np.ceil(H / box_size))
    n_cols = int(np.ceil(W / box_size))

    N_boxes = 0
    for i in range(n_rows):
        # compute the y–slice for this row of boxes
        y0 = i * box_size
        y1 = min((i+1) * box_size, H)

        for j in range(n_cols):
            # compute the x–slice for this column of boxes
            x0 = j * box_size
            x1 = min((j+1) * box_size, W)

            # if any point in this sub-array is True, count the box
            if np.any(field[y0:y1, x0:x1]):
                N_boxes += 1

    return N_boxes, box_size  

def count_boxes_through_scales(fractal, scales):
    count_through_scales = []
    int_scales = []
    for scale in scales:
        count, int_scale = count_boxes(fractal, scale)
        count_through_scales.append(count)
        int_scales.append(int_scale)
    return np.array(count_through_scales), np.array(int_scales)

def log_count_for_j_scales(fractal, scales= list(range(1,50))):
    count_through_scales, integer_scales = count_boxes_through_scales(fractal, scales)
    log_count = np.log(count_through_scales)
    log_scales = np.log(integer_scales)
    return log_count, log_scales

def compute_dim(X, min_idx, max_idx):
    log_count, log_scales = log_count_for_j_scales(X)
    H, V = linear_regression(log_scales[min_idx:max_idx], log_count[min_idx:max_idx])

    return -H, V, log_count, log_scales

def fractal_dim_folder(folder, title_plot=None):
    '''
    Computes the fractal dimension for all the .npz files contained in a specific folder
    '''
    hist_video= []
    for file in os.listdir(folder):
        img = np.load(folder+file,allow_pickle=True)
        img[img<1e-3]=-1
        img[img>=1e-3]= 1
        hist_video.append(img)
    
    estimate_fractal_dimension(hist_video, title_plot)
