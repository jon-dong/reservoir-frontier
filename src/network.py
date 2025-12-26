import torch
import numpy as np
from tqdm import trange

from . import linop


class Network(torch.nn.Module):
    def __init__(
        self,
        width,
        depth,
        mode: str = "rand",
        W_bias=None,
        config_linop: dict | None = None,
        config_resid: dict | None = None,
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
        self.n_linops = config_linop.get("n_linops", 1)
        if mode == "rand":
            self.linops = [
                linop.Random(size=width, dtype=dtype, device=device)
                for _ in range(self.n_linops)
            ]
        elif mode == "struct":
            n_layers = config_linop.get("n_layers", 2)
            mags = config_linop.get("mags", ["unit", "unit"])
            osr = config_linop.get("osr", 1.0)
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
                for _ in range(self.n_linops)
            ]
        elif mode == "conv":
            kernel_size = config_linop.get("kernel_size", width)
            self.linops = [
                linop.RandomConvolution(
                    shape=(width,),
                    kernel_size=kernel_size,
                    dtype=dtype,
                    device=device,
                )
                for _ in range(self.n_linops)
            ]

        self.resid_span = config_resid.get("resid_span", None)
        self.resid_stride = config_resid.get("resid_stride", None)
        self.layer_outputs = [None] * (self.depth + 1)

    def iter(self, x, b, counter, W_scales=None, b_scales=None):
        """Perform a single layer iteration of the network.

        This method computes one forward pass through a single layer by applying
        a linear operator, adding bias, and passing through an activation function.
        It supports multiple scaling factors for both weights and biases simultaneously,
        producing outputs for all combinations of scales.

        Args:
            x: Input activations of shape (width,) or (n_W_scales, n_b_scales, width).
               If 1D, it will be expanded to 3D.
            b: Bias vector for this layer, shape (width,).
            counter: Index of the linear operator to use (cycles through n_linops).
            W_scales: Scalar or array-like of scaling factors for the weight matrix.
                      If None, defaults to [1.0]. Shape (n_W_scales,).
            b_scales: Scalar or array-like of scaling factors for the bias vector.
                      If None, defaults to [1.0]. Shape (n_b_scales,).

        Returns:
            tuple: (x_new, updated_counter) where:
                - x_new: Activated output of shape (n_W_scales, n_b_scales, width).
                - updated_counter: Next linear operator index (wraps to 0 after n_linops).

        Note:
            The computation is: x_new = activation(W_scales * W @ x + b_scales * b)
            For complex inputs, only the real part is activated and scaled by sqrt(2).
            For 'rand' mode, output is additionally normalized by 1/sqrt(width).
        """
        if W_scales is None:
            W_scales = torch.tensor([1.0]).to(self.device)
        if b_scales is None:
            b_scales = torch.tensor([1.0]).to(self.device)
        W_scales = torch.tensor(W_scales).to(self.dtype).to(self.device)
        b_scales = torch.tensor(b_scales).to(self.dtype).to(self.device)

        if x.ndim == 1:
            x = x[None, None, :]
        Wx = self.linops[counter].apply(x)
        W_scales = W_scales[:, None, None]
        b = b[None, None, :]
        b_scales = b_scales[None, :, None]
        z = W_scales * Wx + b_scales * b

        if torch.is_complex(z):
            x_new = self.activation(z.real) * np.sqrt(2)
        else:
            x_new = self.activation(z)

        counter += 1
        if counter == self.n_linops:
            counter = 0

        if self.mode == "rand":
            x_new /= np.sqrt(self.width)
            return x_new, counter
        elif self.mode in ["struct", "conv"]:
            return x_new, counter
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
        """Execute a full forward pass through the entire network.

        This method propagates the input through all layers of the network, applying
        linear transformations, biases, and activations at each step. It supports
        parallel analysis by computing outputs for all combinations of weight
        and bias scaling factors simultaneously.

        Args:
            x: Initial input vector of shape (width,).
            bs: Bias vectors for all layers, shape (depth, width). These will be
                transformed by W_bias before use.
            W_scales: List of scaling factors for weight matrices. The forward pass
                      computes outputs for each scale. Default [1.0].
            b_scales: List of scaling factors for bias vectors. The forward pass
                      computes outputs for each scale. Default [1.0].
            n_save_last: Number of final layer outputs to save. If n_save_last=1,
                         only the final output is saved. If n_save_last=k, the last
                         k layer outputs are saved. Default 1.
            normalize: If True, applies z-score normalization (subtract mean, divide
                       by std) to activations after each layer. Default False.
            resid_start: Layer index at which to start adding residual connections.
                         Only used if resid_span is configured. Default 1.

        Returns:
            torch.Tensor: Saved outputs of shape (n_save_last, n_W_scales, n_b_scales, width).
                          Contains the activations from the last n_save_last layers,
                          computed for all combinations of W_scales and b_scales.

        Note:
            - If resid_span and resid_stride are configured, residual connections
              are added: x[i] += x[i - resid_span] every resid_stride layers after
              layer resid_start.
            - The linear operators cycle through the n_linops operators during iteration.
            - Progress is displayed via tqdm progress bar.
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

        counter = 0
        for i in trange(1, self.depth + 1):
            x, counter = self.iter(
                x,
                bs[i - 1, :],
                counter,
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
