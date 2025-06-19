## Various module prototypes

from __future__ import annotations
from typing import Union, Tuple
from collections import OrderedDict
from soket.nn import Module
import soket.ops as oops
import soket


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
