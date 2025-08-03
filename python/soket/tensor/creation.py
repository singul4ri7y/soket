## OPERATIONS WHICH CREATES A NEW TENSOR ##

from __future__ import annotations
from typing import Tuple, Optional
from .tensor import Tensor
from soket.backend import Device
from soket.backend.device import default_device
from soket.dtype import DType, default_datatype


# Tensor creation functions decorator
def creation_decor(func) -> function:
    def wrapper(
        *shape,
        dtype: Optional[DType] = default_datatype,
        device: Optional[Device] = None,
        **kwargs
    ):
        if len(shape) > 0 and isinstance(shape[0], (list, tuple)):
            shape = shape[0]

        for i, s in enumerate(shape):
            assert isinstance(s, int), f'Invalid shape {s} on index {i}'

        device = default_device() if not device else device
        assert isinstance(device, Device), f'Invalid device {device}'

        return func(shape, device=device, dtype=str(DType(dtype)), **kwargs)

    return wrapper

@creation_decor
def rand(
    shape: Tuple[int],
    low: float = 0.0,
    high: float = 1.0,
    device: Device = None,
    dtype: DType = None,
    **kwargs
) -> Tensor:
    """ Returns Tensor filled with samples from uniform distribution of interval [low, high). """

    array = device.rand(shape, dtype=dtype) * (high - low) + low
    return Tensor(array, **kwargs)


@creation_decor
def randn(
    shape: Tuple[int],
    mean: float = 0.0,
    std: float = 1.0,
    device: Device = None,
    dtype: DType = None,
    **kwargs
) -> Tensor:
    """
    Returns a tensor filled with random samples from a normal distribution of
    `mean` and `std` variance.
    """

    array = device.rand(shape, dtype=dtype) * std + mean
    return Tensor(array, **kwargs)


@creation_decor
def randb(
    shape: Tuple[int],
    prob: float = 0.5,
    device: Device = None,
    dtype: DType = None,
    **kwargs
) -> Tensor:
    """ Returns tensor filled with samples from binomial distribution of probability `prob`. """

    return Tensor(device.rand(shape, dtype=dtype) <= prob, **kwargs)


@creation_decor
def full(
    shape: Tuple[int],
    fill: any = 1.0,
    device: Device = None,
    dtype: DType = None,
    **kwargs
) -> Tensor:
    """ Returns a tensor filled with `fill` value of given shape. """

    return Tensor(device.full(shape, fill, dtype=dtype), **kwargs)


@creation_decor
def zeroes(
    shape: Tuple[int],
    device: Device = None,
    dtype: DType = None,
    **kwargs
) -> Tensor:
    """ Returns a tensor filled with zeros of given shape. """

    return Tensor(device.zeros(shape, dtype=dtype), **kwargs)


@creation_decor
def zeros_like(
    tensor: Tensor,
    device: Device = None,
    dtype: DType = None,
    **kwargs
) -> Tensor:
    """ Returns a tensor of the same shape as given tensor but filled with zeros. """

    return zeroes(tensor.shape, device=device, dtype=dtype, **kwargs)


@creation_decor
def ones(
    shape: Tuple[int],
    device: Device = None,
    dtype: DType = None,
    **kwargs
) -> Tensor:
    """ Returns a tensor of given shape filled with ones. """

    return Tensor(device.ones(shape, dtype=dtype), **kwargs)


@creation_decor
def ones_like(
    tensor: Tensor,
    device: Device = None,
    dtype: DType = None,
    **kwargs
) -> Tensor:
    """ Returns a tensor of the same shape as given tensor but filled with ones. """

    return ones(tensor.shape, device=device, dtype=dtype, **kwargs)


def one_hot(
    tensor: Tensor,
    num_classes: int = -1,
    device: Optional[Device] = None,
    dtype: Optional[DType] = default_datatype,
    **kwargs
) -> Tensor:
    """ Returns an one-hot encoded tensor. """

    assert isinstance(tensor, Tensor), f'Invalid tensor input {tensor}'
    dtype = str(DType(dtype)) if dtype else None

    device = default_device() if not device else device
    assert isinstance(device, Device), f'Invalid device {device}'

    array = tensor.compute_cached_data()

    # Get maximum number of classes
    if num_classes == -1:
        num_classes = array.max()

    return Tensor(device.one_hot(array, num_classes=num_classes,
        dtype=str(DType(dtype))), **kwargs)
