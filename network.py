import torch
import numpy as np

from linop import LinOp


class Network(torch.nn.Module):
    def __init__(self, layer_size=100, 
                 device='cpu'):
        super().__init__()
        self.layer_size = layer_size
        self.linop = LinOp().to(device)
        self.f = torch.erf

    def forward(self, initial_state, biases, 
                n_history=10, weight_scale=1., bias_scale=1.):
        n_layers = biases.shape[0]
        result = torch.zeros(n_history, self.layer_size)
        current_state = initial_state
        for i_layer in range(n_layers):
            current_state = self.single_iter(current_state, biases[i_layer, :], 
                                             weight_scale=weight_scale,
                                             bias_scale=bias_scale)
            if i_layer >= n_layers-n_history:
                result[i_layer-n_layers+n_history, :] = current_state
        return result

    def forward_parallel(self, initial_states, biases,
                         n_history=10, weight_scale_list=[1.], bias_scale=1.):
        n_weight_scale = len(weight_scale_list)
        n_layers = biases.shape[0]
        result = torch.zeros(n_weight_scale, n_history, self.layer_size)
        current_states = initial_states
        for i_layer in range(n_layers):
            current_states = self.parallel_iter(current_states, biases[i_layer, :], 
                                               weight_scale_list=weight_scale_list,
                                               bias_scale=bias_scale)
            if i_layer >= n_layers-n_history:
                result[:, i_layer-n_layers+n_history, :] = current_states
        return result

    def single_iter(self, layer_state, bias, 
                    weight_scale=1., bias_scale=1.):
        return self.f(
            weight_scale * self.linop.apply(layer_state) +
            bias_scale * bias.to(self.device)
        ) / np.sqrt(self.layer_size)

    def parallel_iter(self, layer_states, bias, 
                      weight_scale_list=[1.], bias_scale=1.):
        n_weight_scale = len(weight_scale_list)
        weight_scales = torch.tensor(weight_scale_list).to(self.device).unsqueeze(1)
        bias_term = bias.repeat(n_weight_scale, 1).to(self.device)
        return self.f(
            weight_scales * self.linop.apply(layer_states) +
            bias_scale * bias_term
        ) / np.sqrt(self.layer_size)
    
    