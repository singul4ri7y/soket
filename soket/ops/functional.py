from __future__ import annotations
from typing import Tuple
from soket.ops import Log, Exp, LogSumExp

def log(tensor: Tensor) -> Tensor:
    return Log()(tensor)


def exp(tensor: Tensor) -> Tensor:
    return Exp()(tensor)


def logsumexp(
    tensor: Tensor,
    *axes: Tuple[int],
    keepdims: bool = False
) -> Tensor:
    axsiz = len(axes)
    if axsiz > 0 and isinstance(axes[0], (tuple, list)):
        axes = axes[0]

    return LogSumExp(None if axsiz == 0 else tuple(axes), keepdims)(tensor)