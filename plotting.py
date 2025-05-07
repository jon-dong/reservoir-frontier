import numpy as np
import torch
import matplotlib.pyplot as plt
import porespy as ps
import os
from tqdm import tqdm
from Reservoir import CustomReservoir
from utils import *



def final_plot_threshold(folder, threshold_list_number = 32):
  fields = [] 
  level = 0
  thresh_list = np.logspace(-5, 0, threshold_list_number)
  fig, axs = plt.subplots(1,len(os.listdir(folder)), sharey='row', figsize=[20,7])
  final_spore_list = [0 for i in range(len(os.listdir(folder)))]
  final_dim_edge_list = [0 for i in range(len(os.listdir(folder)))]
  max_thresholds = [0 for i in range(len(os.listdir(folder)))]
  for idx, file in enumerate(sorted(os.listdir(folder))):
      img = np.load(folder+file,allow_pickle=True)
      fields.append(img.copy())
      #dim_list = []
      dim_list_edge = []
      # dim_list_spore = []
      #dim_list_edge_spore = []
      max_dim = 0
      
      for threshold in thresh_list:

          #field_of_zeros = hist_video[idx] >= threshold

          #field_of_zeros = fractals[idx]

          #threshold = 1e-1
          field_of_zeros = (np.abs(fields[idx]) >= level) & (np.abs(fields[idx]) <= (level +threshold))
          
          signed_field = np.ones(field_of_zeros.shape)
          signed_field[field_of_zeros]*= -1
          edges= extract_edges(signed_field)


          min_idx = 0
          max_idx = -1
          #H, log_count, log_scales = compute_dim(field_of_zeros, min_idx, max_idx)
          H_edges, log_count_edges, log_scales_edges = compute_dim(edges, min_idx, max_idx)

          #print(f"Slope (H): {H}, Slope edges (H): {H_edges}")#, Intercept (V): {V}")
          # dim_spore = estimate_fractal_dimension([signed_field])
          # ret_spore_zero = ps.metrics.boxcount(field_of_zeros)
          ret_spore_edge = ps.metrics.boxcount(edges)
          #print(f'dim utils : {dim_spore} \t dim  : {np.median(ret_spore_zero.slope)} \t dim edge : {np.median(ret_spore_edge.slope)}')
          if H_edges > max_dim:
             max_dim = H_edges
             max_thresholds[idx] = threshold
             final_dim_edge_list[idx] = [H_edges, log_count_edges, log_scales_edges]
             final_spore_list[idx] = ret_spore_edge
             
          dim_list_edge.append(H_edges)
          # dim_list_edge.append([H_edges, log_count_edges, log_scales_edges])
          # # dim_list_spore.append(np.median(ret_spore_zero.slope))
          # dim_list_edge_spore.append()
  
      axs[idx].scatter(thresh_list, dim_list_edge)
      axs[idx].set_title(file.replace(folder.replace("/",""),"").replace("_HR.npy",""))
      axs[idx].grid(True)
      axs[idx].set_xscale('log')

  #Title can be adjusted
  fig.suptitle(r'dim $\partial L_{\leq \varepsilon}$ by lin reg',fontsize=16)

  fig.savefig(f'{folder.replace("/","")}_threshold_plot.pdf')

  return final_dim_edge_list, final_spore_list, max_thresholds

def fractal_dim_convergence_plots(fields, threshold_list):


  for (field, threshold) in zip(fields,threshold_list):
    field_of_zeros = np.abs(field) <= threshold

    signed_field = np.ones(field_of_zeros.shape)
    signed_field[field_of_zeros]*= -1
    edges = extract_edges(signed_field)


    min_idx = 0
    max_idx = -1
    #H, log_count, log_scales = compute_dim(field_of_zeros, min_idx, max_idx)
    H_edges, log_count_edges, log_scales_edges = compute_dim(edges, min_idx, max_idx)

    fig, axs = plt.subplots(1,2, figsize= (12,6))


    axs[0].plot(log_scales_edges, log_count_edges)
    ret_spore_zero = ps.metrics.boxcount(edges)

    axs[1].axhline(H_edges)
    axs[1].plot(ret_spore_zero.size, ret_spore_zero.slope)#[:-3]

def fractal_dim_convergence_plots2(folder, final_dim_edge_list, final_spore_list):
  _, axs = plt.subplots(1,2, figsize= (12,6))
  for (dim_edge, ret_spore_zero) in zip(final_dim_edge_list, final_spore_list):
      
    
    H_edges, log_count_edges, log_scales_edges = dim_edge

    axs[0].plot(log_scales_edges, log_count_edges)

    axs[1].axhline(H_edges)
    axs[1].plot(ret_spore_zero.size, ret_spore_zero.slope)
  plt.savefig(f'{folder.replace("/","")}_convergence_plot.pdf')
  return


if __name__=='__main__':
   #fractal_dim_folder('250130stability_frontier_data/', title_plot='prova')
  folder = '250130/'
  final_dim_edge_list, final_spore_list, max_thresholds = final_plot_threshold(folder, threshold_list_number = 4)
  fractal_dim_convergence_plots2(folder, final_dim_edge_list, final_spore_list)