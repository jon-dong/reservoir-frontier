import torch
import numpy as np
from tqdm import trange

from . import linop


class Network(torch.nn.Module):
    def __init__(
        self,
        width,
        depth,
        W_bias,
        mode: str = "rand",
        n_linops: int = 1,
        resid_span: int | None = None,  # length of the residual connection
        resid_stride: int | None = None,  # distance between two residual connections
        config: dict | None = None,
        dtype=torch.float32,
        device="cpu",
    ):
        super().__init__()

        self.dtype = dtype
        self.device = device

        self.width = width
        self.depth = depth
        self.W_bias = W_bias.to(device, dtype)
        self.activation = torch.erf

        self.mode = mode
        if mode == "rand":
            self.linops = [
                linop.Random(size=width, dtype=dtype, device=device)
                for _ in range(n_linops)
            ]
        elif mode == "struct":
            n_layers = config.get("n_layers", 2)
            mags = config.get("mags", ["unit", "unit"])
            osr = config.get("osr", 1.0)
            assert len(mags) == n_layers or n_layers - len(mags) == 0.5, (
                "Number of mags must be equal to n_layers or n_layers - len(mags) == 0.5"
            )
            self.linops = [
                linop.StructuredRandom(
                    shape=(width,),
                    n_layers=n_layers,
                    mags=mags,
                    osr=osr,
                    dtype=dtype,
                    device=device,
                )
                for _ in range(n_linops)
            ]
        elif mode == "conv":
            kernel_size = config.get("kernel_size", width)
            self.linops = [
                linop.RandomConvolution(
                    shape=(width,),
                    kernel_size=kernel_size,
                    dtype=dtype,
                    device=device,
                )
                for _ in range(n_linops)
            ]
        self.n_linops = n_linops
        self.counter = 0

        self.resid_span = resid_span
        self.resid_stride = resid_stride
        self.layer_outputs = [None] * (self.depth + 1)

    def iter(self, x, b, W_scales=None, b_scales=None):
        """Iteration on one layer.

        Output is of shape (n_W_scales, n_b_scales, width)
        """
        if W_scales is None:
            W_scales = torch.tensor([1.0]).to(self.device)
        if b_scales is None:
            b_scales = torch.tensor([1.0]).to(self.device)
        W_scales = torch.tensor(W_scales).to(self.dtype).to(self.device)
        b_scales = torch.tensor(b_scales).to(self.dtype).to(self.device)

        if x.ndim == 1:
            x = x[None, None, :]
        Wx = self.linops[self.counter].apply(x)
        W_scales = W_scales[:, None, None]
        b = b[None, None, :]
        b_scales = b_scales[None, :, None]
        z = W_scales * Wx + b_scales * b

        if torch.is_complex(z):
            x_new = self.activation(z.real) * np.sqrt(2)
        else:
            x_new = self.activation(z)

        self.counter += 1
        if self.counter == self.n_linops:
            self.counter = 0

        if self.mode == "rand":
            x_new /= np.sqrt(self.width)
            return x_new
        elif self.mode in ["struct", "conv"]:
            return x_new
        else:
            raise ValueError("Invalid mode")

    def forward(
        self,
        x,
        bs,
        W_scales=[1.0],
        b_scales=[1.0],
        n_save_last=1,
        normalize=False,
        resid_start=1,
    ):
        """Forward pass on the entire network.

        Input x is of shape (width,), output is of shape (n_save_last, n_W_scales, n_b_scales, width)
        """
        assert x.shape == (self.width,), (
            f"Input has incorrect shape {x.shape}, expected ({self.width},)"
        )
        assert bs.shape == (self.depth, self.width), (
            f"Biases has incorrect shape {bs.shape}, expected ({self.depth}, {self.width})"
        )

        n_W_scales = len(W_scales)
        n_b_scales = len(b_scales)

        bs = torch.einsum(
            "ij,nj -> ni",
            self.W_bias,
            bs,
        )  # * for reproducibility, can also pass unormalized Gaussian biases

        x = x.repeat(n_W_scales, n_b_scales, 1).to(self.device, self.dtype)
        outputs = torch.zeros(n_save_last, n_W_scales, n_b_scales, self.width).to(
            self.device
        )
        if self.resid_span is not None:
            self.layer_outputs[0] = x  # length depth + 1

        for i in trange(1, self.depth + 1):
            x = self.iter(
                x,
                bs[i - 1, :],
                W_scales=W_scales,
                b_scales=b_scales,
            )
            # * res connection with pre-activation
            if self.resid_span is not None:
                if i > resid_start and (i - resid_start) % self.resid_stride == 0:
                    x += self.layer_outputs[i - self.resid_span]
                self.layer_outputs[i] = x
            if normalize:
                # current = torch.nn.functional.normalize(current, p=2, dim=1)
                x = (x - torch.mean(x, dim=1, keepdim=True)) / (
                    torch.std(x, dim=1, keepdim=True) + 1e-10
                )
            if i > (self.depth - n_save_last):
                outputs[i - 1 - self.depth + n_save_last] = x
        return outputs

    def stability_test(
        self,
        x1=None,
        x2=None,
        bs=None,
        W_scales=None,
        b_scales=None,
        mode="independent",
        noise_level=0.01,
        normalize=False,
        n_save_last=1,
    ):
        """
        Stability test on the same input and different reservoir scales

        Follows the distance between the reservoir states through time, whether they converge to the same trajectory

        Returns:
        dist: shape (n_scales, n_history)
        """
        bs = bs.to(self.device, self.dtype)

        if mode == "independent":
            if x1 is None and x2 is None:
                x1 = torch.randn(self.width).to(self.device, self.dtype)
                x1 = x1 / torch.norm(x1)
                x2 = torch.randn(self.width).to(self.device, self.dtype)
                x2 = x2 / torch.norm(x2)
        elif mode == "sensitivity":
            x1 = torch.randn(self.width).to(self.device, self.dtype)
            epsilon = noise_level * torch.randn(self.width).to(self.device, self.dtype)
            x2 = x1 + epsilon

            x1 = x1 / torch.norm(x1)
            x2 = x2 / torch.norm(x2)

        outputs1 = self.forward(
            x1,
            bs,
            W_scales=W_scales,
            b_scales=b_scales,
            normalize=normalize,
            n_save_last=n_save_last,
        )

        self.counter = 0
        outputs2 = self.forward(
            x2,
            bs,
            W_scales=W_scales,
            b_scales=b_scales,
            normalize=normalize,
            n_save_last=n_save_last,
        )

        return torch.sum((outputs1 - outputs2) ** 2, dim=-1)
