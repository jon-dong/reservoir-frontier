import torch


class LinOp:
    def __init__(self, state_size, W_res=None, mode='random',dtype=torch.float64,device='cpu'):
        self.dtype = dtype
        self.device = device
        if mode == 'random':
            if W_res is None:
                self.matrix = torch.randn(state_size, state_size).to(self.dtype).to(self.device)
            else:
                self.matrix = W_res.to(self.dtype).to(self.device)
        else:
            raise NotImplementedError

    def apply(self, x):
        """ perform the linear operator on the input x.
        
        x can be either a single vector or a batch of vectors.
        """
        return torch.einsum('ab, ...b -> ...a', self.matrix, x)