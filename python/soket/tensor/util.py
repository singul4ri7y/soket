from __future__ import annotations
from typing import List, Tuple, Sequence
from .tensor import Tensor
from soket.backend import NDArray


def stack(tensors: Sequence[Tensor], axis: int = 0) -> Tensor:
    """ Stacks (concatenates) a sequence of tensors over an axis. """

    assert isinstance(tensors, (list, tuple)), f'Expected a sequence of tensors'
    for t in tensors:
        assert isinstance(t, Tensor), f'Expected all to be tensors in given sequence'

    return Tensor(
        NDArray.stack([x.compute_cached_data() for x in tensors], axis=axis),
    )
