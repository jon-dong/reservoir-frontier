import torch
import numpy as np

import linop


class Network(torch.nn.Module):
    def __init__(
        self,
        input_size,
        state_size,
        input_scale,
        depth,
        W_in,
        W_res,
        n_linops=1,
        n_layers=None,
        mode="random",
        dtype=torch.float64,
        device="cpu",
    ):
        super().__init__()
        self.input_size = input_size
        self.state_size = state_size
        self.input_scale = input_scale
        self.W_in = W_in.to(dtype).to(device)
        if W_res is not None:
            self.W_res = W_res.to(dtype).to(device)
        self.depth = depth
        self.dtype = dtype
        self.device = device
        self.mode = mode
        if mode == 'random':
            self.linops = [linop.Random(
                state_size=state_size, dtype=dtype, device=device
            ) for _ in range(n_linops)]
        elif mode == 'structured_random':
            self.linops = [ linop.StructuredRandom(
                shape=(state_size,), n_layers=n_layers, dtype=dtype, device=device
            ) for _ in range(n_linops) ]
        self.n_linops = n_linops
        self.counter = 0
        self.f = torch.erf

    def iter_single(self, input, bias, weight_scale=1.0, bias_scale=1.0):
        """single forward for a single state scale on a single input.

        returns:
        res: shape (state_size)
        """
        aft_act = self.f(
            weight_scale * self.linops[self.counter].apply(input) + bias_scale * bias.to(self.device)
        )
        self.counter += 1
        if self.counter == self.n_linops:
            self.counter = 0
        if self.mode == 'random':
            return aft_act / np.sqrt(self.state_size)
        elif self.mode == 'structured_random':
            return aft_act
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

        pre_act = torch.einsum("n, ni -> ni", weight_scales, self.linops[self.counter].apply(inputs)) + bias_scale * biases
        if torch.is_complex(pre_act):
            aft_act = self.f(pre_act.real)
        else:
            aft_act = self.f(pre_act)
        self.counter += 1
        if self.counter == self.n_linops:
            self.counter = 0
        if self.mode == 'random':
            return aft_act / np.sqrt(self.state_size)
        elif self.mode == 'structured_random':
            return aft_act
        else:
            raise ValueError("Invalid mode")

    def forward_single(
        self, sequence, state=None, n_history=10, weight_scale=1.0, bias_scale=None
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
        assert sequence.shape[0] == self.depth, "Depth mismatch"
        assert sequence.shape[1] == self.input_size, "Input size mismatch"

        n_layers = sequence.shape[0]
        sequence = sequence.to(self.device)

        biases = torch.einsum(
            "ij,nj -> ni",
            self.W_in,
            sequence,
        )

        if state is not None:
            assert state.shape[0] == self.state_size, "State size mismatch"
            current = state
        else:
            current = torch.zeros(self.state_size).to(self.device)
        if bias_scale is None:
            bias_scale = self.input_scale

        res = torch.zeros(n_history, self.state_size)
        for i in range(n_layers):
            current = self.iter_single(
                current, biases[i], weight_scale=weight_scale, bias_scale=bias_scale
            )
            if i >= n_layers - n_history:
                res[i - n_layers + n_history, :] = current
        return res

    def forward_parallel(
        self,
        sequence,
        state,
        weight_scales=[1.0],
        bias_scale=None,
        n_history=20,
    ):
        """forward pass on multiple state scales for a single input.

        params:
        n_history: number of the last states to keep, to be used for averagin
        returns:
        res: shape (n_scales, n_history, state_size)"""
        assert sequence.shape[0] == self.depth, "Depth mismatch"
        assert sequence.shape[1] == self.input_size, "Input size mismatch"

        n_scales = len(weight_scales)

        biases = torch.einsum(
            "ij,nj -> ni",
            self.W_in,
            sequence,
        )

        if state is not None:
            assert (
                state.shape[0] == self.state_size
            ), "State size mismatch, expected {}, got {}".format(
                self.state_size, state.shape[0]
            )
            current = state.repeat(n_scales, 1).to(self.dtype).to(self.device)
        else:
            current = (
                torch.zeros(n_scales, self.state_size).to(self.dtype).to(self.device)
            )
        if bias_scale is None:
            bias_scale = self.input_scale
        if n_history is None:
            n_history = self.depth
        res = torch.zeros(n_scales, n_history, self.state_size).to(self.device)
        for i in range(self.depth):
            current = self.iter_parallel(
                current,
                biases[i, :],
                weight_scales=weight_scales,
                bias_scale=bias_scale,
            )
            if i >= (self.depth - n_history):
                res[:, i - self.depth + n_history, :] = current
        return res

    def stability_test(self, sequence, weight_scales, state1=None, state2=None):
        """
        Stability test on the same input and different reservoir scales

        Follows the distance between the reservoir states through time, whether they converge to the same trajectory

        Returns:
        dist: shape (n_scales, n_history)
        """
        n_scales = len(weight_scales)
        sequence = sequence.to(self.device).to(self.dtype)

        if state1 is None:
            state1 = torch.randn(self.state_size).to(self.dtype).to(self.device)
            state1 = state1 / torch.norm(state1)
            state1 = state1.repeat(n_scales, 1)
        if state2 is None:
            state2 = torch.randn(self.state_size).to(self.dtype).to(self.device)
            state2 = state2 / torch.norm(state2)
            state2 = state2.repeat(n_scales, 1)
        states1 = self.forward_parallel(sequence, state1, weight_scales=weight_scales)
        self.counter = 0
        states2 = self.forward_parallel(sequence, state2, weight_scales=weight_scales)
        return torch.sum((states1 - states2) ** 2, dim=2)
