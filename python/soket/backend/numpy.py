""" CPU backend implementation using NumPy """

from typing import Tuple, List
import numpy

NDArray = numpy.ndarray

# TODO: Probably should be changed later
array_api = numpy

class Device:
    """ Baseclass for all devices. """

class CPUDevice(Device):
    """ Represents a CPU device, where computations are performed on data
    sitting in RAM by CPU.

    """

    def __repr__(self) -> str:
        return "soket.CPU()"

    def __hash__(self) -> int:
        return self.__repr__().__hash__()

    def __eq__(self, other) -> bool:
        return isinstance(other, CPUDevice)

    def enabled(self) -> bool:
        return True

    # TODO: Maybe use decorators?

    def zeros(self, *shape: Tuple[int], dtype: str = 'float32') -> NDArray:
        # If the shape is passed as a tuple or list
        if len(shape) > 0 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return numpy.zeroes(shape, dtype=dtype)

    def ones(self, *shape: Tuple[int], dtype: str = 'float32') -> NDArray:
        # If the shape is passed as a tuple or list
        if len(shape) > 0 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return numpy.ones(shape, dtype=dtype)

    def randn(self, *shape: Tuple[int], dtype: str = 'float32') -> NDArray:
        # If the shape is passed as a tuple or list
        if len(shape) > 0 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return numpy.random.randn(*shape).astype(dtype)

    def rand(self, *shape: Tuple[int], dtype: str = 'float32') -> NDArray:
        # If the shape is passed as a tuple or list
        if len(shape) > 0 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return numpy.random.rand(*shape).astype(dtype)

    def one_hot(self, n: int, i: NDArray, dtype: str = 'float32') -> NDArray:
        return numpy.eye(n, dtype=dtype)[i]

    def empty(self, *shape, dtype: str = 'float32') -> NDArray:
        # If the shape is passed as a tuple or list
        if len(shape) > 0 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return numpy.empty(shape, dtype=dtype)

    def full(self, *shape, fill: float = 0., dtype: str = 'float32') -> NDArray:
        # If the shape is passed as a tuple or list
        if len(shape) > 0 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return numpy.full(shape, fill, dtype=dtype)

def cpu() -> Device:
    """ Return CPU device. """
    return CPUDevice()

def default_device() -> Device:
    return cpu()

def all_devices() -> List[Device]:
    """ Return a list of all available devices. """
    return [cpu()]

