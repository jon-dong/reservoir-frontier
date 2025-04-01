import numpy as np
import torch
import matplotlib.pyplot as plt
import porespy as ps
import os
from tqdm import tqdm
from Reservoir import CustomReservoir

def erf_frontier(res_scale):
    return np.sqrt(4*res_scale**4/np.pi**2 - 1/(4) - 2*res_scale**2/np.pi*np.arcsin((16*res_scale**4-np.pi**2)/(16*res_scale**4+np.pi**2))+1e-6)


def stability_test(res_size=100, input_size=100, input_len=100, resolution=20, constant_input=False,
                   res_scale_bounds=[0, 3], input_scale_bounds=[0, 2], 
                   average=10, device='cpu', seed=0):
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
        input_data = torch.randn(input_len, input_size).to(device)
        for i in range(input_len):  # normalize input at each timestep
            input_data[i, :] = input_data[i, :] / torch.norm(input_data[i, :])
    else:
        input_data = torch.randn(input_size).to(device)
        input_data = input_data / torch.norm(input_data)
        input_data = input_data.repeat(input_len, 1)

    res_scale_list = np.linspace(res_scale_bounds[0], res_scale_bounds[1], num=resolution)
    input_scale_list = np.linspace(input_scale_bounds[0], input_scale_bounds[1], num=resolution)
    final_metric = torch.zeros(resolution, resolution)

    # Initializel reservoir and initial states
    W_in = torch.randn(res_size, input_size).to(device)
    W_res = torch.randn(res_size, res_size).to(device)
    initial_state1 = torch.randn(res_size).to(device) / np.sqrt(res_size)
    initial_state1 = initial_state1 / torch.norm(initial_state1)
    initial_state2 = torch.randn(res_size).to(device) / np.sqrt(res_size)
    initial_state2 = initial_state2 / torch.norm(initial_state2)

    for (i_in, input_scale) in tqdm(enumerate(input_scale_list)):
        RC = CustomReservoir(f="erf", input_size=input_size, res_size=res_size,
                             W_res=W_res, W_in=W_in,
                             input_scale=input_scale, device=device)
        rc_metric = RC.stability_test(input_data, res_scale_list, initial_state1=initial_state1, initial_state2=initial_state2)
        final_metric[:, i_in] = torch.mean(rc_metric[:, -average:], dim=1)
    return final_metric

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

if __name__=='__main__':
   fractal_dim_folder('250130stability_frontier_data/', title_plot='prova')