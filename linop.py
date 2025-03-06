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

    def sample(self, shape: tuple[int, ...]) -> np.ndarray:
        # compute the maximum value of the pdf if not yet computed
        if self.max_pdf is None:
            self.max_pdf = np.max(
                self.pdf(np.linspace(self.min_supp + 1e-8, self.max_supp - 1e-8, 10000))
            )

        samples = []
        while len(samples) < np.prod(shape):
            x = np.random.uniform(self.min_supp, self.max_supp, size=1)
            y = np.random.uniform(0, self.max_pdf, size=1)
            if y < self.pdf(x):
                samples.append(x)
        return np.array(samples).reshape(shape)


class MarchenkoPastur(Distribution):
    def __init__(self, m: int, n: int, sigma=None):
        self.m = m
        self.n = n
        # when oversampling ratio is 1, the distribution has min support at 0, leading to a very high peak near 0 and numerical issues.
        self.gamma = n / m
        if sigma is not None:
            self.sigma = sigma
        else:
            # automatically set sigma to make E[|x|^2] = 1
            # self.sigma = (1+self.gamma)**(-0.25)
            self.sigma = 1
        self.lamb = m / n
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

    def sample(self, shape, include_zero=False, equisampling=False) -> np.ndarray:
        """using acceptance-rejection sampling if oversampling ratio is more than 1, otherwise using the eigenvalues sampled from a real matrix"""
        if self.m < self.n:
            # there will be n - m zero eigenvalues, the rest nonzero eigenvalues follow the Marchenko-Pastur distribution
            if include_zero is True:
                n_zeros = int(np.prod(shape) / self.n * (self.n - self.m))
                nonzeros = super().sample((np.prod(shape) - n_zeros,))
                zeros = np.zeros(n_zeros)
                return np.random.permutation(np.concatenate((nonzeros, zeros))).reshape(
                    shape
                )
            elif equisampling:
                sub_distribution = MarchenkoPastur(self.n, self.n)
                return sub_distribution.sample(shape)
            else:
                return super().sample(shape)
        elif self.m == self.n:
            # compute the eigenvalues from a real matrix and use it as the samples
            X = 1 / np.sqrt(self.m) * th.randn((self.m, self.n), dtype=th.cfloat)
            eigenvalues_X, _ = th.linalg.eig(X.conj().T @ X)
            return np.array(eigenvalues_X).reshape(shape)
        else:
            return super().sample(shape)

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
        mags=["marchenko", "unit"],
        oversampling_ratio=1.5,
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
                self.diagonals.append(
                    MarchenkoPastur(oversampling_ratio * shape[0], shape[0]).sample(shape)
                    * Rademacher(shape, dtype, device)
                )
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
