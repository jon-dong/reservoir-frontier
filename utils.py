import numpy as np
import torch

def erf_frontier(res_scale):
    return np.sqrt(4*res_scale**4/np.pi**2 - 1/(4) - 2*res_scale**2/np.pi*np.arcsin((16*res_scale**4-np.pi**2)/(16*res_scale**4+np.pi**2))+1e-6)