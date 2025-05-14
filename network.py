import torch
import numpy as np

import linop


class Network(torch.nn.Module):
    def __init__(
        self,
        width,
        depth,
        bias_scale,
        W_bias,
        n_hist=20,
        n_linops=1,
        n_layers=None,
        mode="rand",
        residual_length=None,  # the length of the residual connection
        residual_interval=None,  # the distance between two residual connections
        kernel_size=None,
        mags=["unit", "unit"],
        osr=1.5,
        dtype=torch.float32,
        device="cpu",
    ):
        super().__init__()

        self.dtype = dtype
        self.device = device

        self.width = width
        self.depth = depth
        self.W_bias = W_bias.to(device, dtype)
        self.bias_scale = bias_scale
        self.mode = mode
        self.n_hist = n_hist
        if mode == "rand":
            self.linops = [
                linop.Random(size=width, dtype=dtype, device=device)
                for _ in range(n_linops)
            ]
        elif mode == "struct":
            assert len(mags) == n_layers or n_layers - len(mags) == 0.5, (
                "Number of mags must be equal to n_layers or n_layers - len(mags) == 0.5"
            )
            self.linops = [
                np.sqrt(2) * linop.StructuredRandom(
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

        self.residual_length = residual_length
        self.residual_interval = residual_interval
        self.hist_states = [None] * (self.depth + 1)

        self.activation = torch.erf

    def iter_single(self, input, bias, weight_scale=1.0, bias_scale=1.0):
        """single forward for a single state scale on a single input.

        returns:
        res: shape (state_size)
        """
        output = self.activation(
            weight_scale * torch.real(self.linops[self.counter].apply(input.view(1, 1, -1)))
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

    def forward_single(
        self, input, biases, weight_scale=1.0, bias_scale=None
    ):
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
        start=100,
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

        self.hist_states[0] = curr # length depth + 1
        outputs = torch.zeros(n_scales, self.n_hist, self.width).to(self.device)

        for i in range(1, self.depth+1):
            curr = self.iter_parallel(
                curr,
                biases[i-1, :],
                weight_scales=weight_scales,
                bias_scale=bias_scale,
            )
            # * res connection with pre-activation
            if self.residual_length is not None:
                if i > start and (i-start) % self.residual_interval == 0:
                    # print(f'adding residual connection from layer {i + 1 - self.residual_length} to layer {i + 1}')
                    curr += self.hist_states[i - self.residual_length]
                # add new states
                self.hist_states[i] = curr
            if normalize:
                # current = torch.nn.functional.normalize(current, p=2, dim=1)
                curr = (curr - torch.mean(curr, dim=1, keepdim=True))/ (torch.std(curr, dim=1, keepdim=True) + 1e-5)
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
            input1 = torch.randn(self.width).to(self.device,self.dtype)
            epsilon = noise_level * torch.randn(self.width).to(self.device, self.dtype)
            input2 = input1 + epsilon

            input1 = input1 / torch.norm(input1)
            input2 = input2 / torch.norm(input2)

        self.counter = 0
        outputs1 = self.forward_parallel(input1, biases,  weight_scales=weight_scales, normalize=normalize)

        self.counter = 0
        outputs2 = self.forward_parallel(input2, biases, weight_scales=weight_scales, normalize=normalize)

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
            input1 = torch.randn(self.width).to(self.device,self.dtype)
            epsilon = noise_level * torch.randn(self.width).to(self.device, self.dtype)
            input2 = input1 + epsilon

            input1 = input1 / torch.norm(input1)
            input2 = input2 / torch.norm(input2)

        self.counter = 0
        outputs1 = self.forward_single(input1, biases,  weight_scale=weight_scale)#, normalize=normalize)

        self.counter = 0
        outputs2 = self.forward_single(input2, biases, weight_scale=weight_scale)#, normalize=normalize)

        return (outputs1 - outputs2) ** 2
        # return torch.sum((outputs1 - outputs2) ** 2, dim=1)