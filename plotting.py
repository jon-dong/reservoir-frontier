import numpy as np
import torch
import matplotlib.pyplot as plt
import porespy as ps
import os
from tqdm import tqdm
from Reservoir import CustomReservoir
from utils import *
from matplotlib.gridspec import GridSpec
import matplotlib

def zooming_plot(folder):
  
  plt.figure()
  hist_video= []

  for file in sorted(os.listdir(folder)):
      img = np.load(folder+file,allow_pickle=True)
      hist_video.append(img.copy())

  threshold = 1e-5
  resolution = img.shape[0]
  

  bias_scale_min = [1.62, 1.73, 1.777, 1.785]
  bias_scale_max = [1.92, 1.83, 1.802, 1.795]
  weight_scale_min = [1, 1.1, 1.137, 1.145]
  weight_scale_max = [1.3, 1.2, 1.162, 1.155]

  fig, axs = plt.subplots(2,2)
  idx_bin = [[0,0],[0,1],[1,0],[1,1]]
  for idx, img in enumerate(hist_video):
      img[img<threshold]= threshold
      ylab = np.linspace(bias_scale_min[idx], bias_scale_max[idx], num=2)
      xlab = np.linspace(weight_scale_min[idx], weight_scale_max[idx], num=2)
      indXx = np.linspace(0, resolution-1, num=xlab.shape[0]).astype(int)
      indXy = np.linspace(0, resolution-1, num=ylab.shape[0]).astype(int)
      axs[idx_bin[idx][0],idx_bin[idx][1]].imshow(img.T,norm=matplotlib.colors.LogNorm(vmin= 1e-4, vmax = 1))
      axs[idx_bin[idx][0],idx_bin[idx][1]].grid(False)
      axs[idx_bin[idx][0],idx_bin[idx][1]].set_xticks(indXx)
      axs[idx_bin[idx][0],idx_bin[idx][1]].set_xticklabels(xlab)
      axs[idx_bin[idx][0],idx_bin[idx][1]].set_yticks(indXy)
      axs[idx_bin[idx][0],idx_bin[idx][1]].set_yticklabels(ylab)


  fig.supxlabel('Weight scale')
  fig.supylabel('Bias scale')
  fig.suptitle('Asymptotic stability metric\nfor $f=$erf')
  fig.tight_layout()
  plt.savefig(f'{folder.replace("/","")}_zoomings.pdf')

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
  color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
  for (dim_edge, ret_spore_zero,c) in zip(final_dim_edge_list, final_spore_list,color):
      
    
    H_edges, log_count_edges, log_scales_edges = dim_edge

    axs[0].plot(log_scales_edges, log_count_edges)

    axs[1].axhline(H_edges, c=c)
    axs[1].plot(ret_spore_zero.size, ret_spore_zero.slope, marker='d', c=c)
    #axs[1].scatter(ret_spore_zero.size, ret_spore_zero.slope, marker='x', c=c)
  plt.savefig(f'{folder.replace("/","")}_convergence_plot.pdf')
  return

def final_plot_threshold_all(folder, threshold_list_number = 32):
  fields = [] 
  level = 0
  thresh_list = np.logspace(-4, 0, threshold_list_number)
  fig = plt.figure(layout="constrained")
  gs = GridSpec(4, 8, figure=fig)
  axs = []
  #fig, axs = plt.subplots(1,len(os.listdir(folder)), sharey='row', figsize=[20,7])
  final_spore_list = [0 for i in range(len(os.listdir(folder)))]
  final_dim_edge_list = [0 for i in range(len(os.listdir(folder)))]
  max_thresholds = [0 for i in range(len(os.listdir(folder)))]
  titles = ['original', 'zoom1', 'zoom2', 'zoom3']
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
      axs.append(fig.add_subplot(gs[:2, 2*idx:2*(idx+1)]))
      axs[-1].scatter(thresh_list, dim_list_edge)
      axs[-1].set_title(titles[idx], fontsize=10)
      axs[-1].grid(True)
      axs[-1].set_xscale('log')
      xticks = np.logspace(-4, 0, 3)
      xlabels = ['$10^{-4}$','$10^{-2}$','$1$']
      axs[-1].set_xticks(xticks, labels=xlabels)
  color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
  labels = ['original', 'zoom1', 'zoom2', 'zoom3']
  axs.append(fig.add_subplot(gs[2:, :4]))
  axs.append(fig.add_subplot(gs[2:, 4:]))
  for (dim_edge, ret_spore_zero,c, label) in zip(final_dim_edge_list, final_spore_list,color,labels):
      
    
    H_edges, log_count_edges, log_scales_edges = dim_edge

    axs[-2].plot(log_scales_edges, log_count_edges)
    axs[-1].axhline(H_edges, c=c)
    axs[-1].plot(ret_spore_zero.size, ret_spore_zero.slope,label=label, marker='d', c=c)
    axs[-1].set_xscale('log')
    axs[-1].legend()
    #axs[1].scatter(ret_spore_zero.size, ret_spore_zero.slope, marker='x', c=c)
  
  #Title can be adjusted
  fig.suptitle(r'dim $\partial L_{\leq \varepsilon}$ by lin reg',fontsize=16)

  fig.savefig(f'{folder.replace("/","")}_summary.pdf')

  return

if __name__=='__main__':
   #fractal_dim_folder('250130stability_frontier_data/', title_plot='prova')
  folder = '250130/'
  #final_dim_edge_list, final_spore_list, max_thresholds = 
  #final_plot_threshold_all(folder, threshold_list_number = 31)
  #fractal_dim_convergence_plots2(folder, final_dim_edge_list, final_spore_list)
  zooming_plot(folder)