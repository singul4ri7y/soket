from __future__ import annotations
from typing import List, Tuple
from soket import Tensor
from soket.backend.numpy import array_api


def stack(tensors: List[Tensor] | Tuple[Tensor], axis: int = 0) -> Tensor:
    """ Stacks (concatenates) a sequence of tensors over an axis. """
    return Tensor(
        array_api.stack([x.compute_cached_data() for x in tensors], axis=axis),
        device=tensors[0].device,
        dtype=tensors[0].dtype
    )