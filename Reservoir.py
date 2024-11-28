import numpy as np
import torch

use_cuda = torch.cuda.is_available()
default_device = torch.device("cuda:2" if use_cuda else "cpu")


class CustomReservoir(torch.nn.Module):
    """
    Implements a minimal Reservoir Computing algorithm for stability analysis.
    :param input_size: dimension of the input
    :param res_size: number of units in the reservoir
    :param input_scale: scale of the input-to-reservoir matrix
    :param res_scale: scale of the reservoir weight matrix
    :param f: activation function of the reservoir
    :param seed: random seed for reproducibility of the results
    """

    def __init__(self, input_size, res_size,
                 input_scale=1.0, res_scale=1.0, 
                 W_res=None, W_in=None,
                 f='tanh',
                 seed=None, device=default_device):
        super().__init__()

        # Parameter initialization
        self.input_size = input_size
        self.res_size = res_size
        self.input_scale = input_scale
        self.res_scale = res_scale
#         self.seed = seed
        self.device = device

        # Weights generation
#         torch.manual_seed(self.seed)
        if W_in is None:
          self.W_in = torch.randn(res_size, input_size).to(self.device)
        else:
          self.W_in = W_in
        if W_res is None:
          self.W_res = torch.randn(res_size, res_size).to(self.device)
        else:
          self.W_res = W_res

        # Activation function
        if f == 'erf':
            self.f = torch.erf
        elif f == 'heaviside':
            self.f = lambda x: 1 * (x > 0)
        elif f == 'sign':
            self.f = torch.sign
        elif f == 'linear':
            self.f = lambda x: x
        elif f == 'relu':
            self.f = torch.relu
        elif f == 'tanh':
            self.f = torch.tanh
        elif f == 'heaviside':
            self.f = torch.heaviside

    def forward(self, input_data, initial_state=None):
        """
        Compute the reservoir states for the given sequence
        :param input_data: input sequence of shape (seq_len, input_size)
        :param initial_state: initial reservoir state at t=0 (res_size, )
        :return: successive reservoir states (seq_len+1, res_size)
        """
        input_data = input_data.to(self.device)
        seq_len = input_data.shape[0]
        states = torch.zeros((seq_len+1, self.res_size)).to(self.device)  
        # will contain the reservoir states
        if initial_state is not None:
            states[0, :] = initial_state

        for i in range(seq_len):
            states[i+1, :] = self.f(
                    self.input_scale * self.W_in @ input_data[i, :] +
                    self.res_scale * self.W_res @ states[i, :]
                ) / np.sqrt(self.res_size)
        return states
    
    def forward_parallel(self, input_data, res_scale_list, initial_state=None):
        """
        Compute the reservoir states for the given sequence
        :param input_data: input sequence of shape (seq_len, input_size)
        :param initial_state: initial reservoir state at t=0 (res_size, )
        :return: successive reservoir states (seq_len+1, res_size)
        """
        n_res_scale = len(res_scale_list)
        input_data = input_data.to(self.device)
        seq_len = input_data.shape[0]
        states = torch.zeros((n_res_scale, seq_len+1, self.res_size)).to(self.device)  
        # will contain the reservoir states
        if initial_state is not None:
            states[0, :] = initial_state

        res_scale_tensor = torch.tensor(res_scale_list).to(self.device).unsqueeze(1)
        for i in range(seq_len):
            input_contribution = self.input_scale * self.W_in @ input_data[i, :]
            input_contribution = input_contribution.repeat(n_res_scale, 1)
            res_contributions = res_scale_tensor * (states[:, i, :] @ self.W_res)
            states[:, i+1, :] = self.f(
                    input_contribution +
                    res_contributions
                ) / np.sqrt(self.res_size)
        return states
        
    def stability_test(self, input_data, res_scale_list, initial_state1=None, initial_state2=None):
        """
        Stability test: Iterate the reservoir for the same input data and two different initial state
        Follows the distance between the reservoir states through time, whether they converge to the same trajectory
        """
        n_res_scale = len(res_scale_list)
        if initial_state1 is None:
            initial_state1 = torch.randn(self.res_size).to(self.device) / np.sqrt(self.res_size)
            initial_state1 = initial_state1 / torch.norm(initial_state1)
            initial_state1 = initial_state1.repeat(n_res_scale, 1)
        if initial_state2 is None:
            initial_state2 = torch.randn(self.res_size).to(self.device) / np.sqrt(self.res_size)
            initial_state2 = initial_state2 / torch.norm(initial_state2)
            initial_state2 = initial_state2.repeat(n_res_scale, 1)
        states1 = self.forward_parallel(input_data, res_scale_list, initial_state1)
        states2 = self.forward_parallel(input_data, res_scale_list, initial_state2)
        return torch.sum((states1 - states2)**2, dim=2)
