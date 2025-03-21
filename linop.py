import math
from abc import ABC, abstractmethod

import numpy as np
import torch as th

from base_linop import LinOp


class Distribution(ABC):
    def __init__(self):
        self.min_supp: float
        self.max_supp: float
        self.max_pdf = None

    @abstractmethod
    def pdf(self, x) -> np.ndarray:
        pass

    def sample(self, shape: int|tuple[int, ...]) -> np.ndarray:
        if isinstance(shape, int):
            shape = (shape,)
        
        # compute the maximum value of the pdf if not yet computed
        if self.max_pdf is None:
            self.max_pdf = np.max(
                self.pdf(np.linspace(self.min_supp + 1e-8, self.max_supp - 1e-8, 10000))
            )

        n_samples = np.prod(shape)
        samples = np.empty(0)
        while samples.shape[0] < n_samples:
            n_need = int(n_samples - samples.shape)
            x = np.random.uniform(self.min_supp, self.max_supp, size=n_need)
            y = np.random.uniform(0, self.max_pdf, size=n_need)
            accepted = x[np.where(y < self.pdf(x))]
            if accepted.shape[0] >= n_need:
                samples = np.append(samples, accepted[:n_need])
                break
            else:
                samples = np.append(samples, accepted)
        return samples.reshape(shape)


class MarchenkoPastur(Distribution):
    """
    Marchenko-Pastur distribution.

    It describes the asymptotic eigenvalue distribution of the matrix X = 1/sqrt(m) A^T A, where A is a matrix of shape m times n and sampled i.i.d. from a distribution with zero mean and variance sigma^2
    """
    def __init__(self, alpha:float, sigma:float=1.0):
        """
        alpha: oversampling ratio
        sigma: standard deviation of the element distribution
        """
        assert alpha >= 0, 'oversampling ratio must be nonnegative' 
        assert sigma >= 0, 'standard deviation must be nonnegative'
        self.alpha = alpha # oversampling ratio
        self.gamma = 1/alpha
        self.sigma = 1
        self.min_supp = self.sigma**2 * (1 - math.sqrt(self.gamma)) ** 2
        self.max_supp = self.sigma**2 * (1 + math.sqrt(self.gamma)) ** 2
        super().__init__()

    def pdf(self, x: np.ndarray) -> np.ndarray:
        assert (x >= self.min_supp).all() and (x <= self.max_supp).all(), (
            "x is out of the support of the distribution"
        )
        return np.sqrt((self.max_supp - x) * (x - self.min_supp)) / (
            2 * np.pi * self.sigma**2 * self.gamma * x
        )

    def sample(self,
               shape,
               normalized=False,
               include_zero=True,
               ) -> np.ndarray:
        """using acceptance-rejection sampling if oversampling ratio is more than 1, otherwise using the eigenvalues sampled from a real matrix"""
        n_samples = np.prod(shape)
        if self.alpha < 1.0:
            # undersampling
            # there will be zero eigenvalues, the rest nonzero eigenvalues follow the pdf
            if include_zero is True:
                n_zeros = int(n_samples * (1 - self.alpha))
                nonzeros = super().sample((n_samples - n_zeros,))
                zeros = np.zeros(n_zeros)
                samples = np.random.permutation(np.concatenate((nonzeros, zeros)))
            else:
                samples = super().sample((n_samples,))
        elif self.alpha == 1.0:
            # equisampling
            #! The distribution has min support at 0, leading to a very high peak near 0 and difficulty to sample from acceptance-rejection sampling
            #! Instead, we directly eigenvalue decompose a matrix to get the eigenvalues 
            X = 1 / np.sqrt(n_samples) * th.randn((n_samples, n_samples), dtype=th.cfloat)
            samples, _ = th.linalg.eig(X.conj().T @ X)
        else:
            # oversampling
            samples = super().sample(shape)
        if normalized:
            # normalize the samples such that E[x^2] = 1
            samples = samples / np.sqrt(1+self.gamma) / (self.sigma**2)
        return samples.reshape(shape)

    def mean(self):
        return self.sigma**2

    def var(self):
        return self.gamma * self.sigma**4


class Random(LinOp):
    def __init__(self, state_size, W_res=None, dtype=th.float64, device="cpu"):
        self.dtype = dtype
        self.device = device
        if W_res is None:
            self.matrix = (
                th.randn(state_size, state_size).to(self.dtype).to(self.device)
            )
        else:
            self.matrix = W_res.to(self.dtype).to(self.device)

    def apply(self, x):
        """perform the linear operator on the input x.

        x can be either a single vector or a batch of vectors.
        """
        res = th.einsum("ab, ...b -> ...a", self.matrix, x)
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
    def __init__(
        self,
        shape: tuple,
        n_layers: int | float,
        mags=["unit", "unit"],
        osr=1.5,
        dtype=th.float64,
        device=th.device("cpu"),
    ):
        assert len(mags) == n_layers or n_layers - len(mags) == 0.5, "Number of mags must be equal to n_layers or n_layers - len(mags) == 0.5"

        self.in_shape = shape
        self.out_shape = shape
        self.n_layers = n_layers
        self.diagonals = []
        for mag in mags:
            if mag == "unit":
                self.diagonals.append(
                    Rademacher(shape, dtype, device)
                )
            elif mag == "marchenko":
                diagonal = Rademacher(shape, dtype, device)
                diagonal.values = th.tensor(MarchenkoPastur(osr).sample(shape, normalized=True)).to(dtype).to(device) * diagonal.values
                self.diagonals.append(diagonal)
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


class RandomConvolution(LinOp):
    def __init__(
        self,
        shape: tuple,
        kernel_size: int,
        dtype=th.float64,
        device=th.device("cpu"),
    ):
        self.in_shape = shape
        self.out_shape = shape
        self.kernel_size = kernel_size
        self.kernel = th.nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size, padding=kernel_size // 2).to(dtype).to(device)
        self.kernel.weight.data = th.randn(1, 1, kernel_size).to(dtype).to(device)/np.sqrt(kernel_size)
        self.dtype = dtype
        self.device = device

    def apply(self, x):
        x = x.unsqueeze(1)
        with th.no_grad():  # Disable autograd for Conv1d
            x = self.kernel(x)
        x = x.squeeze(1)
        return x
