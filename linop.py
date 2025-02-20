from abc import abstractmethod
import math
import torch as th

## Base class
class LinOp:
    def __init__(self):
        self.in_shape: tuple
        self.out_shape: tuple

    @abstractmethod
    def apply(self, x: th.Tensor) -> th.Tensor:
        pass

    @abstractmethod
    def applyT(self, y: th.Tensor) -> th.Tensor:
        pass

    def __add__(self, other):
        if isinstance(other, LinOp):
            return Sum(self, other)
        else:
            raise NameError(
                "Summing scalar and LinOp objects does not result in a linear operator."
            )

    def __radd__(self, other):
        if isinstance(other, LinOp):
            return Sum(self, other)
        else:
            raise NameError(
                "Summing scalar and LinOp objects does not result in a linear operator."
            )

    def __sub__(self, other):
        if isinstance(other, LinOp):
            return Diff(self, other)
        else:
            raise NameError(
                "Subtracting scalar and LinOp objects does not result in a linear operator."
            )

    def __rsub__(self, other):
        if isinstance(other, LinOp):
            return Diff(self, other)
        else:
            raise NameError(
                "Subtracting scalar and LinOp objects does not result in a linear operator."
            )

    def __mul__(self, other):
        if isinstance(other, LinOp):
            raise NameError(
                "Multiplying two LinOp objects does not result in a linear operator."
            )
        else:
            return ScalarMul(self, other)

    def __rmul__(self, other):
        if isinstance(other, LinOp):
            raise NameError(
                "Multiplying two LinOp objects does not result in a linear operator."
            )
        else:
            return ScalarMul(self, other)

    def __matmul__[T: (LinOp, th.Tensor)](self, other: T) -> T:
        if isinstance(other, LinOp):
            return Composition(self, other)
        return self.apply(other)

    @property
    def T(self):
        return Transpose(self)


## Utils classes
class Composition(LinOp):
    def __init__(self, LinOp1: LinOp, LinOp2: LinOp):
        self.LinOp1 = LinOp1
        self.LinOp2 = LinOp2
        self.in_shape = LinOp1.in_shape if LinOp2.in_shape == (-1,) else LinOp2.in_shape
        self.out_shape = (
            LinOp2.out_shape if LinOp1.out_shape == (-1,) else LinOp1.out_shape
        )

    def apply(self, x):
        return self.LinOp1.apply(self.LinOp2.apply(x))

    def applyT(self, y):
        return self.LinOp2.applyT(self.LinOp1.applyT(y))


class Sum(LinOp):
    def __init__(self, LinOp1: LinOp, LinOp2: LinOp):
        self.LinOp1 = LinOp1
        self.LinOp2 = LinOp2
        self.in_shape = (
            LinOp1.in_shape if LinOp1.in_shape > LinOp2.in_shape else LinOp2.in_shape
        )
        self.out_shape = (
            LinOp1.out_shape
            if LinOp1.out_shape > LinOp2.out_shape
            else LinOp2.out_shape
        )

    def apply(self, x):
        return self.LinOp1.apply(x) + self.LinOp2.apply(x)

    def applyT(self, y):
        return self.LinOp2.applyT(y) + self.LinOp1.applyT(y)


class Diff(LinOp):
    def __init__(self, LinOp1: LinOp, LinOp2: LinOp):
        self.LinOp1 = LinOp1
        self.LinOp2 = LinOp2
        self.in_shape = (
            LinOp1.in_shape if LinOp1.in_shape > LinOp2.in_shape else LinOp2.in_shape
        )
        self.out_shape = (
            LinOp1.out_shape
            if LinOp1.out_shape > LinOp2.out_shape
            else LinOp2.out_shape
        )

    def apply(self, x):
        return self.LinOp1.apply(x) - self.LinOp2.apply(x)

    def applyT(self, y):
        return self.LinOp1.applyT(y) - self.LinOp2.applyT(y)


class ScalarMul(LinOp):
    def __init__(self, LinOp: LinOp, other):
        self.LinOp = LinOp
        self.scalar = other
        self.in_shape = LinOp.in_shape
        self.out_shape = LinOp.out_shape

    def apply(self, x):
        return self.LinOp.apply(x) * self.scalar

    def applyT(self, y):
        return self.LinOp.applyT(y) * self.scalar


class Transpose(LinOp):
    def __init__(self, LinOpT: LinOp):
        self.LinOpT = LinOpT
        self.in_shape = LinOpT.out_shape
        self.out_shape = LinOpT.in_shape

    def apply(self, x):
        return self.LinOpT.applyT(x)

    def applyT(self, y):
        return self.LinOpT.apply(y)

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
        return th.einsum('ab, ...b -> ...a', self.matrix, x)

class Rademacher(LinOp):
    def __init__(self, shape, dtype, device):
        self.in_shape = shape
        self.out_shape = shape
        # generate a tensor with each element following the rademacher distribution
        self.values = th.randint(0, 2, shape).to(dtype).to(device)
        self.values[self.values == 0] = -1
    
    def apply(self, x):
        assert x.shape == self.in_shape, f"Input shape {x.shape} does not match the expected shape {self.in_shape}"
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
    def __init__(self, shape:tuple, n_layers:int, dtype=th.float64, device=th.device('cpu')):
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
        def forward(x):
            if self.n_layers - math.floor(self.n_layers) > 0:
                return th.fft.fft(x, norm="ortho")
            for i in range(math.floor(self.n_layers)):
                x = self.diagonals[i].apply(x)
                x = th.fft.fft(x, norm="ortho")
            return x
        if len(x.shape) == len(self.in_shape):
            return forward(x)
        elif len(x.shape) == len(self.in_shape)+1:
            return th.stack([forward(x_i) for x_i in x])
        else:
            raise ValueError(f"Input tensor has unexpected shape {x.shape}")