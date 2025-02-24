from abc import abstractmethod
import math
import time
import torch as th

from base_linop import LinOp

class Random(LinOp):
    def __init__(self, state_size, W_res=None, dtype=th.float64,device='cpu'):
        self.dtype = dtype
        self.device = device
        if W_res is None:
            self.matrix = th.randn(state_size, state_size).to(self.dtype).to(self.device)
        else:
            self.matrix = W_res.to(self.dtype).to(self.device)

    def apply(self, x):
        """ perform the linear operator on the input x.
        
        x can be either a single vector or a batch of vectors.
        """
        res = th.einsum('ab, ...b -> ...a', self.matrix, x)
        return res

class Rademacher(LinOp):
    def __init__(self, shape, dtype, device):
        self.in_shape = shape
        self.out_shape = shape
        # generate a tensor with each element following the rademacher distribution
        self.values = th.randint(0, 2, shape).to(dtype).to(device)
        self.values[self.values == 0] = -1
    
    def apply(self, x):
        return self.values * x

class Fft(LinOp):
    def __init__(self):
        self.in_shape = (-1,)
        self.out_shape = (-1,)

    def apply(self, x):
        return th.fft.fft(x, norm="ortho")

    def applyT(self, y):
        return th.fft.ifft(y, norm="ortho")

class Identity(LinOp):
    def __init__(self):
        super().__init__()
        self.in_shape = (-1,)
        self.out_shape = (-1,)
    
    def apply(self, x):
        return x

class StructuredRandom(LinOp):
    def __init__(self, shape:tuple, n_layers:int|float, dtype=th.float64, device=th.device('cpu')):
        self.in_shape = shape
        self.out_shape = shape
        self.n_layers = n_layers
        self.diagonals = [Rademacher(shape,dtype,device) for _ in range(math.floor(n_layers))]
        self.dtype = dtype
        self.device = device
        # if n_layers - math.floor(n_layers) > 0:
        #     self.model = Fft()
        # else:
        #     self.model = Identity()
        # for i in range(math.floor(n_layers)):
        #     self.model = self.diagonals[i] @ self.model
        #     self.model = Fft() @ self.model
    
    def apply(self, x):
        # benchmark running time after every operation
        if self.n_layers - math.floor(self.n_layers) > 0:
            x = th.fft.fft(x, norm="ortho")
        for i in range(math.floor(self.n_layers)):
            x = self.diagonals[i].apply(x)
            x = th.fft.fft(x, norm="ortho")
        return x