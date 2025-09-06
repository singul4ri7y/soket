# For NN initialization

import soket
from typing import Tuple

def xavier_normal(tensor, gain=1.0):
    """ Note: `fan_in` and `fan_out` is calculated assuming the weight is
    not transposed. In other words, it is assumed that for Linear layers
    the propagation is done is this manner: X @ W.
    """

    fan_in, fan_out = tensor.shape[-2:]
    std_sq = gain * gain * (2 / (fan_in + fan_out))

    tensor.data = soket.randn(tensor.shape, mean=0.0, std=std_sq,
        device=tensor.device, dtype=tensor.dtype)


def xavier_uniform(tensor, gain=1.0):
    """ Note: `fan_in` and `fan_out` is calculated assuming the weight is
    not transposed. In other words, it is assumed that for Linear layers
    the propagation is done is this manner: X @ W.
    """

    fan_in, fan_out = tensor.shape[-2:]
    a = gain * gain * (6 / (fan_in + fan_out))

    tensor.data = soket.rand(tensor.shape, low=-a, high=a,
        device=tensor.device, dtype=tensor.dtype)


def _kaiming_prologue(shape: Tuple, mode='fan_in', nonlinearity='relu') -> Tuple[float, int]:
    """ Note: `fan_in` and `fan_out` is calculated assuming the weight is
    not transposed. In other words, it is assumed that for Linear layers
    the propagation is done is this manner: X @ W.

    Available modes: 'fan_in' and 'fan_out'
    """

    # These non-linearity gains are measured by analyzing the plot of activation norm
    # and gradient norm vs no of layers in DNN.
    nonlinearity_gain = {
        'linear': 1.0,
        'identity': 1.0,
        'conv': 1.0,
        'sigmoid': 1.0,
        'tanh': 1.6666,
        'relu': 1.4142
    }

    assert mode == 'fan_in' or mode == 'fan_out', 'Invalid mode'
    assert nonlinearity_gain.get(nonlinearity) != None, 'Invalid nonlinearity'

    fan = {
        'fan_out': shape[-1],
        'fan_in': shape[-2]
    }

    return nonlinearity_gain[nonlinearity], fan[mode]


def kaiming_normal(tensor, mode='fan_in', nonlinearity='relu'):
    """ Kaiming normal distribution initialization. """

    gain, fan_mode = _kaiming_prologue(tensor.shape, mode, nonlinearity)
    std_sq = gain * gain / fan_mode

    tensor.data = soket.randn(tensor.shape, mean=0.0, std=std_sq,
        device=tensor.device, dtype=tensor.dtype)

def kaiming_uniform(tensor, mode='fan_in', nonlinearity='relu'):
    """ Kaiming uniform distribution initialization. """

    gain, fan_mode = _kaiming_prologue(tensor.shape, mode, nonlinearity)
    bound = gain * gain * 3 / fan_mode

    tensor.data = soket.rand(tensor.shape, low=-bound, high=bound,
        device=tensor.device, dtype=tensor.dtype)