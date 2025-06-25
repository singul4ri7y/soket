## Various module prototypes

from __future__ import annotations
from typing import Union, Tuple, List
from collections import OrderedDict
from soket.nn import Module
import soket.ops as oops
import soket
import math


class Identity(Module):
    def forward(self, X):
        return X


class Linear(Module):
    def __init__(
        self, feature_in: int, feature_out: int,
        device=None, bias=True, dtype='float32'
    ):
        super().__init__()

        self.feature_in = feature_in
        self.feature_out = feature_out

        # Store weight
        self.weight = soket.zeros(feature_in, feature_out, dtype=dtype,
            device=device, requires_grad=True)

        # Store bias if needed
        self.bias = soket.zeros(feature_out, dtype=dtype, device=device,     \
            requires_grad=True) if bias else None

    def forward(self, X: soket.Tensor) -> soket.Tensor:
        Z = X @ self.weight

        if self.bias is not None:
            Z += self.bias

        return Z


class ReLU(Module):
    def forward(self, X: soket.Tensor) -> soket.Tensor:
        return oops.ReLU()(X)


class Sequential(Module):
    def __init__(self, *modules: Union[Tuple[Module], dict, OrderedDict]):
        super().__init__()

        self.modules_ = []
        if len(modules) > 0:
            if isinstance(modules[0], (dict, OrderedDict)):
                for v in modules[0].values():
                    if isinstance(v, Module):
                        self.modules_.append(v)
            elif isinstance(modules[0], (tuple, list)):
                self.modules_ += list(modules[0])
            else:
                self.modules_ += list(modules)

    def forward(self, X: soket.Tensor) -> soket.Tensor:
        Z = X
        for m in self.modules_:
            Z = m(Z)

        return Z


class SoftmaxCrossEntropyLoss(Module):
    def __init__(
        self,
        *mean_axes: Tuple[int],
        reduction: str = 'mean'
    ):
        super().__init__()

        if len(mean_axes) == 0:
            mean_axes = (-1,)    # Always mean over the columns
        elif isinstance(mean_axes[0], (tuple, list)):
            mean_axes = mean_axes[0]

        self.mean_axes = mean_axes

        assert reduction == 'mean' or reduction == 'none' or reduction == 'sum', \
            'Invalid reduction type'

        self.reduction = reduction

    def forward(self, Z: soket.Tensor, y: soket.Tensor) -> soket.Tensor:
        assert len(Z.shape) > 1, 'Logit tensor shape has to be atleast 2D'
        return oops.SoftmaxCrossEntropy(self.mean_axes, y, Z.shape[-1],
            self.reduction)(Z)


class BatchNorm1d(Module):
    """ Applies Batch Normalization over a 2D or 3D input. """

    def __init__(
        self,
        num_features: int,
        eps: float =1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device=None,
        dtype='float32'
    ):
        super().__init__()

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        ## PARAMETERS ##
        # Can be thought of as weight
        self.gamma = soket.ones(num_features, device=device,
            dtype=dtype, requires_grad=affine)

        # Can be thought of as bias
        self.beta = soket.zeros(num_features, device=device,
            dtype=dtype, requires_grad=affine)

        # Running mean and variance.
        if track_running_stats:
            self.running_mean = soket.Tensor(0.0, device=device, dtype=dtype)
            self.running_var = soket.Tensor(1.0, device=device, dtype=dtype)

    def forward(self, X: soket.Tensor) -> soket.Tensor:
        assert len(X.shape) == 2 or len(X.shape) == 3,    \
            'Batch Normalization (1D) expects the data dimension to be 2D/3D'

        bshape = (self.num_features)
        mean_dim = [ 0 ]
        if len(X.shape) == 3:
            bshape = (1, self.num_features, 1)
            mean_dim.append(2)

        if self.training is True:
            mean = X.mean(mean_dim, keepdims=True)               # Over the mini-batch
            xshift = X - mean
            var = (xshift ** 2).mean(mean_dim, keepdims=True)    # Over the mini-batch

            if self.track_running_stats is True:
                self.running_mean = (1 - self.momentum) * self.running_mean + \
                    self.momentum * mean
                self.running_var = (1 - self.momentum) * self.running_var +   \
                    self.momentum * var

            rvar = (var + self.eps) ** -0.5
            norm = xshift * rvar
        else:
            if self.track_running_stats is True:
                mean = self.running_mean
                var = self.running_var
                xshift = X - mean
            else:
                # If mean and variance was not tracked, use current mini-batch
                # mean and variance
                mean = X.mean(mean_dim, keepdims=True)
                xshift = X - mean
                var = (xshift ** 2).mean(mean_dim, keepdims=True)

            rvar = (var + self.eps) ** -0.5
            norm = xshift * rvar

        return self.gamma.reshape(bshape) * norm + self.beta.reshape(bshape)


class LayerNorm(Module):
    """ Applies Layer Normalization over the final dimensions (normalized_shape)
    of a mini-batch.
    """

    def __init__(
        self,
        *normalized_shape,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        use_beta: bool = True,
        device=None,
        dtype='float32'
    ):
        super().__init__()

        if isinstance(normalized_shape[0], (tuple, list)):
            normalized_shape = normalized_shape[0]

        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.use_beta = use_beta

        # `normalized_shape` is the shape of the trailing dimensions, which needs
        # to be reduced over to find mean and variance.
        self.mean_dim = tuple([-x for x in range(1, len(self.normalized_shape) + 1)])

        if elementwise_affine is True:
            self.gamma = soket.ones(*self.normalized_shape, device=device,
                dtype=dtype, requires_grad=True)

            if use_beta is True:
                self.beta = soket.zeros(*self.normalized_shape, device=device,
                    dtype=dtype, requires_grad=True)

    def forward(self, X: soket.Tensor) -> soket.Tensor:
        assert len(X.shape) >= len(self.normalized_shape) + 1, 'Invalid input dimension'

        mean = X.mean(self.mean_dim, keepdims=True)
        xshift = X - mean
        var = (xshift ** 2).mean(self.mean_dim, keepdims=True)
        rvar = (var + self.eps) ** -0.5

        norm = xshift * rvar
        if self.elementwise_affine is True:
            norm = self.gamma * norm

            if self.use_beta is True:
                norm += self.beta

        return norm


class Flatten(Module):
    def __init__(self, start_dim: int = 0, end_dim: int = -1):
        super().__init__()

        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, X: soket.Tensor) -> soket.Tensor:
        dim_siz = len(X.shape)

        start_dim = dim_siz + self.start_dim if self.start_dim < 0 else    \
            self.start_dim
        assert start_dim >= 0 and start_dim < dim_siz,     \
            'Start dimension index exceeds total dims'

        end_dim = dim_siz + self.end_dim if self.end_dim < 0 else self.end_dim
        assert end_dim >= 0 and end_dim < dim_siz,     \
            'End dimension index exceeds total dims'
        assert end_dim >= start_dim, 'Invalid range'

        # Result will be same as input tensor.
        if start_dim == end_dim:
            return X

        new_dim = X.shape[:start_dim] + (math.prod(X.shape[start_dim:end_dim + 1]),) \
            + X.shape[end_dim + 1:]

        return X.reshape(new_dim)

