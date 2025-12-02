import torch
from tqdm import trange
import numpy as np

from . import linop


class Network(torch.nn.Module):
    def __init__(
        self,
        width,
        depth,
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
                "Number of mags and layers mismatch"
            )
            self.linops = [
                np.sqrt(2)
                * linop.StructuredRandom(
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

    def iter_single(self, input, bias, weight_scale=1.0, bias_scale=1.0):
        """single forward for a single state scale on a single input.

        returns:
        res: shape (state_size)
        """
        output = self.activation(
            weight_scale
            * torch.real(self.linops[self.counter].apply(input.view(1, 1, -1)))
            + bias_scale * bias.to(self.device)
        )
        self.counter += 1
        if self.counter == self.n_linops:
            self.counter = 0
        if self.mode == "rand":
            return output / np.sqrt(self.width)
        elif self.mode in ["struct", "conv"]:
            return output
        else:
            raise ValueError("Invalid mode")

    def iter_parallel(
        self, inputs, bias, weight_scales: list = [1.0], bias_scale: float = 1.0
    ):
        """parallel forward for multiple state scales on a single input.

        returns:
        res: shape (n_scales, state_size)
        """
        n_scales = len(weight_scales)

        weight_scales = torch.tensor(weight_scales).to(self.device)
        bias_scale = torch.tensor(bias_scale).to(self.device)

        biases = bias.repeat(n_scales, 1).to(self.device)

        pre_act = (
            torch.einsum(
                "n, ni -> ni", weight_scales, self.linops[self.counter].apply(inputs)
            )
            + bias_scale * biases
        )

        if torch.is_complex(pre_act):
            output = self.activation(pre_act.real) * np.sqrt(2)
        else:
            output = self.activation(pre_act)

        self.counter += 1
        if self.counter == self.n_linops:
            self.counter = 0

        if self.mode == "rand":
            return output / np.sqrt(self.width)
        elif self.mode in ["struct", "conv"]:
            return output
        else:
            raise ValueError("Invalid mode")

    def iter(self, x, b, W_scales=None, b_scales=None):
        if W_scales is None:
            W_scales = torch.tensor([1.0]).to(self.device)
        if b_scales is None:
            b_scales = torch.tensor([1.0]).to(self.device)

        if x.ndim == 1:
            x = x[None, None, :]
        Wx = self.linops[self.counter].apply(x)
        b = b[None, None, :]
        W_scales = W_scales[:, None, None]
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

    def forward(
        self,
        x,
        bs,
        W_scales=None,
        b_scales=None,
        normalize=False,
        resid_start=1,
        n_save_last=1,
    ):
        """Forward pass on the entire network, input x is of shape (width,), output is of shape (n_hist, n_W_scales, n_b_scales, width)"""
        assert x.shape == (self.width,), (
            f"Input has incorrect shape {x.shape}, expected ({self.width},)"
        )
        assert bs.shape == (self.depth, self.width), (
            f"Biases has incorrect shape {bs.shape}, expected ({self.depth}, {self.width})"
        )

        n_W_scales = len(W_scales)
        n_b_scales = len(b_scales)

        if x.ndim == 1:
            x = x.repeat(n_W_scales, n_b_scales, 1).to(self.device, self.dtype)
        outputs = torch.zeros(n_save_last, n_W_scales, n_b_scales, self.width).to(
            self.device
        )
        if self.resid_span is not None:
            self.layer_outputs[0] = x

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
                x = (x - torch.mean(x, dim=-1, keepdim=True)) / (
                    torch.std(x, dim=-1, keepdim=True) + 1e-10
                )
            if i > (self.depth - n_save_last):
                outputs[i - 1 - self.depth + n_save_last] = x
        return outputs

    def forward_single(self, input, biases, weight_scale=1.0, bias_scale=None):
        """Forward pass on a single scale

        params:
            sequence: shape (n_layers, input_size)
            state: initial state, shape (state_size)
            n_history: number of the last states to keep
            weight_scale: scaling factor for weights
            bias_scale: scaling factor for biases

        returns:
            res: shape (n_history, state_size)
        """
        assert biases.shape[0] == self.depth, "bias length and depth mismatch"
        assert biases.shape[1] == self.width, "bias size and width mismatch"
        assert input.shape[0] == self.width, "input size and width mismatch"

        biases = torch.einsum(
            "ij,nj -> ni",
            self.W_bias,
            biases,
        )

        if bias_scale is None:
            bias_scale = self.bias_scale

        outputs = torch.zeros(self.n_hist, self.width)
        curr = input.to(self.device)

        for i in range(self.depth):
            curr = self.iter_single(
                curr, biases[i], weight_scale=weight_scale, bias_scale=bias_scale
            )
            if i >= self.depth - self.n_hist:
                outputs[i - self.depth + self.n_hist, :] = curr
        return outputs

    def forward_parallel(
        self,
        input,
        biases,
        weight_scales=[1.0],
        bias_scale=None,
        normalize=False,
        start=1,
    ):
        """forward pass on multiple state scales for a single input.

        params:
        n_history: number of the last states to keep, to be used for averaging
        returns:
        res: shape (n_scales, n_history, state_size)"""
        assert biases.shape[0] == self.depth, "Depth mismatch"
        assert biases.shape[1] == self.width, "Input size mismatch"
        assert input.shape[0] == self.width, "input size and width mismatch"

        n_scales = len(weight_scales)

        biases = torch.einsum(
            "ij,nj -> ni",
            self.W_bias,
            biases,
        )

        curr = input.repeat(n_scales, 1).to(self.device, self.dtype)

        if bias_scale is None:
            bias_scale = self.bias_scale

        self.layer_outputs[0] = curr  # length depth + 1
        outputs = torch.zeros(n_scales, self.n_hist, self.width).to(self.device)

        for i in range(1, self.depth + 1):
            curr = self.iter_parallel(
                curr,
                biases[i - 1, :],
                weight_scales=weight_scales,
                bias_scale=bias_scale,
            )
            # * res connection with pre-activation
            if self.resid_span is not None:
                if i > start and (i - start) % self.resid_stride == 0:
                    # print(f'adding residual connection from layer {i + 1 - self.residual_length} to layer {i + 1}')
                    curr += self.layer_outputs[i - self.resid_span]
                # add new states
                self.layer_outputs[i] = curr
            if normalize:
                # current = torch.nn.functional.normalize(current, p=2, dim=1)
                curr = (curr - torch.mean(curr, dim=1, keepdim=True)) / (
                    torch.std(curr, dim=1, keepdim=True) + 1e-10
                )
            if i > (self.depth - self.n_hist):
                outputs[:, i - 1 - self.depth + self.n_hist, :] = curr
        return outputs

    def stability_test(
        self,
        input1=None,
        input2=None,
        biases=None,
        weight_scales=None,
        mode="independent",
        noise_level=0.01,
        normalize=False,
    ):
        """
        Stability test on the same input and different reservoir scales

        Follows the distance between the reservoir states through time, whether they converge to the same trajectory

        Returns:
        dist: shape (n_scales, n_history)
        """
        biases = biases.to(self.device, self.dtype)

        if mode == "independent":
            if input1 is None and input2 is None:
                input1 = torch.randn(self.width).to(self.device, self.dtype)
                input1 = input1 / torch.norm(input1)
                input2 = torch.randn(self.width).to(self.device, self.dtype)
                input2 = input2 / torch.norm(input2)
        elif mode == "sensitivity":
            input1 = torch.randn(self.width).to(self.device, self.dtype)
            epsilon = noise_level * torch.randn(self.width).to(self.device, self.dtype)
            input2 = input1 + epsilon

            input1 = input1 / torch.norm(input1)
            input2 = input2 / torch.norm(input2)

        self.counter = 0
        outputs1 = self.forward_parallel(
            input1, biases, weight_scales=weight_scales, normalize=normalize
        )

        self.counter = 0
        outputs2 = self.forward_parallel(
            input2, biases, weight_scales=weight_scales, normalize=normalize
        )

        return torch.sum((outputs1 - outputs2) ** 2, dim=2)

    def stability_test1d(
        self,
        input1=None,
        input2=None,
        biases=None,
        weight_scale=None,
        mode="independent",
        noise_level=0.01,
        normalize=False,
    ):
        """
        Stability test on the same input and different reservoir scales

        Follows the distance between the reservoir states through time, whether they converge to the same trajectory

        Returns:
        dist: shape (n_scales, n_history)
        """
        biases = biases.to(self.device, self.dtype)

        if mode == "independent":
            if input1 is None and input2 is None:
                input1 = torch.randn(self.width).to(self.device, self.dtype)
                input1 = input1 / torch.norm(input1)
                input2 = torch.randn(self.width).to(self.device, self.dtype)
                input2 = input2 / torch.norm(input2)
        elif mode == "sensitivity":
            input1 = torch.randn(self.width).to(self.device, self.dtype)
            epsilon = noise_level * torch.randn(self.width).to(self.device, self.dtype)
            input2 = input1 + epsilon

            input1 = input1 / torch.norm(input1)
            input2 = input2 / torch.norm(input2)

        self.counter = 0
        outputs1 = self.forward_single(
            input1, biases, weight_scale=weight_scale
        )  # , normalize=normalize)

        self.counter = 0
        outputs2 = self.forward_single(
            input2, biases, weight_scale=weight_scale
        )  # , normalize=normalize)

        return (outputs1 - outputs2) ** 2
        # return torch.sum((outputs1 - outputs2) ** 2, dim=1)
