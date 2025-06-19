from __future__ import annotations
import soket

def rand(*shape, low=0.0, high=1.0, device=None, **kwargs) -> Tensor:
    """ Generate a tensor with random numbers between low and high """

    if len(shape) > 0 and isinstance(shape[0], (tuple, list)):
        shape = shape[0]

    device = soket.cpu() if device is None else device
    array = device.rand(*shape) * (high - low) + low
    return soket.Tensor(array, device=device, **kwargs)


def randn(*shape, mean=0.0, std=1.0, device=None, **kwargs) -> Tensor:
    """ Generate random normal with specified mean and std deviation """

    if len(shape) > 0 and isinstance(shape[0], (tuple, list)):
        shape = shape[0]

    device = soket.cpu() if device is None else device
    array = device.rand(*shape) * std + mean
    return soket.Tensor(array, device=device, **kwargs)

def constant(*shape, c=1.0, device=None, dtype='float32', **kwargs) -> Tensor:
    """ Generate tensor filled with constant """

    if len(shape) > 0 and isinstance(shape[0], (tuple, list)):
        shape = shape[0]

    device = soket.cpu() if device is None else device
    array = device.ones(*shape, dtype=dtype) * c  # note: can change dtype
    return soket.Tensor(array, device=device, **kwargs)

def ones(*shape, **kwargs) -> Tensor:
    """ Generate all-ones tensor """

    return constant(*shape, c=1.0, **kwargs)

def zeros(*shape, **kwargs) -> Tensor:
    """ Generate all-zeros tensor """

    return constant(*shape, c=0.0, **kwargs)

def randb(*shape, p=0.5, device=None, dtype='bool', **kwargs) -> Tensor:
    """ Generate binary random tensor """

    if len(shape) > 0 and isinstance(shape[0], (tuple, list)):
        shape = shape[0]

    device = soket.cpu() if device is None else device
    array = device.rand(*shape) <= p
    return soket.Tensor(array, device=device, dtype=dtype, **kwargs)

def one_hot(n, i, device=None, dtype='float32', requires_grad=False) -> Tensor:
    """ Generate one-hot encoding tensor """

    device = soket.cpu() if device is None else device
    return soket.Tensor(
        device.one_hot(n, i.numpy(), dtype=dtype),
        device=device,
        requires_grad=requires_grad,
    )

def zeros_like(tensor, *, device=None, requires_grad=False) -> Tensor:
    assert isinstance(tensor, soket.Tensor), 'Input has to be Tensor'

    device = device if device else tensor.device
    return zeros(
        *tensor.shape, dtype=tensor.dtype, device=device,
        requires_grad=requires_grad
    )

def ones_like(tensor, *, device=None, requires_grad=False) -> Tensor:
    assert isinstance(tensor, soket.Tensor), 'Input has to be Tensor'

    device = device if device else tensor.device
    return ones(
        *tensor.shape, dtype=tensor.dtype, device=device,
        requires_grad=requires_grad
    )
