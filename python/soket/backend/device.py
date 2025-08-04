from __future__ import annotations
from typing import Tuple, List, Optional
from types import ModuleType
from soket.dtype import default_datatype
from .ndarray import NDArray, ndarray_ensure
from cupy.cuda import Device as cupy_device
import numpy
import cupy


## DEFAULT DATATYPE ##
_def_dt = str(default_datatype)


## NOTE: Writing datatypes as strings is recommended for backend code instead of using soket.DType.

## SUPPORTED DEVICES ##
_supported_devices = [ 'cpu', 'gpu' ]

# If someone wants to be fancy
class DeviceType:
    CPU: str = _supported_devices[0]
    GPU: str = _supported_devices[1]

class Device:
    """ Represents a compute device. """

    _type: str = DeviceType.CPU
    _id: int = 0  # Ignored for CPU device

    # Internals of a device.
    #   Use NumPy for CPU device
    #   Use CuPy for GPU device
    _backend: ModuleType = numpy
    _backend_device: object = None

    def __init__(self, type: str, id: Optional[int] = 0):
        assert isinstance(type, str) and type in _supported_devices, \
            f'Usupported device type {type}'
        assert isinstance(id, int), f'Invalid id {id}'

        self._type = type
        self._id = id

        # Set appropriate backend
        self._backend = cupy if type is DeviceType.GPU else numpy

        # Keep handle to GPU/CUDA device
        if type is DeviceType.GPU:
            self._backend_device = cupy_device(id)

    def __enter__(self) -> Device:
        """ When enter a `with` block. """

        # Save and switch the default device
        global _DEFAULT_DEVICE
        self._previous_default_device = _DEFAULT_DEVICE
        _DEFAULT_DEVICE = self

        if self._backend_device is not None:
            self._backend_device.__enter__()

        return self

    def __exit__(self, exec_type, exec_value, traceback):
        """ When exit a `with` block. """

        # Switch back to previous default device
        global _DEFAULT_DEVICE
        _DEFAULT_DEVICE = self._previous_default_device

        if self._backend_device is not None:
            self._backend_device.__exit__(exec_type, exec_value, traceback)

    def __eq__(self, other: Device) -> bool:
        """ Test equality. """

        if not isinstance(other, Device):
            return False

        is_equal = self._type == other._type
        if self._backend_device is not None:
            is_equal &= self._backend_device == self._backend_device

        return is_equal

    def __ne__(self, other: Device) -> bool:
        """ Test inequality. """

        return not self.__eq__(other)

    def __repr__(self):
        if self._type is DeviceType.GPU:
            return f'<soket GPU Device, id={self._id}>'

        return '<soket CPU Device>'

    ## PROPERTIES ##

    @property
    def type(self) -> DeviceType:
        """ Returns the device type. """

        return self._type

    @property
    def id(self) -> int:
        """ Returns the device ID being used (GPU only). """

        if self._type is DeviceType.GPU:
            return self._id

        return -1

    ## GPU SPECIFIC OPERATIONS ##

    def use(self):
        """ Use current device (GPU only). """

        if self._type is DeviceType.GPU:
            self._backend_device.use()

    def sync(self):
        """
        Synchronize current thread to the device (GPU only). Can be useful for waiting for an
        operation to complete in the GPU side.
        """

        if self._type is DeviceType.GPU:
            self._backend_device.synchronize()

    ## BASIC OPERATIONS EXPECTED FROM THE DEVICE ##

    def rand(self, shape: Tuple[int] | List[int], dtype: str = None) -> NDArray:
        """ Return random values of given shape. """

        # Generate random arrays
        rand_array = self._backend.random.rand(*shape)

        return NDArray(ndarray_ensure(rand_array, self), self, dtype=dtype)

    def zeros(self, shape: Tuple[int] | List[int], dtype: str = None) -> NDArray:
        """ Return NDArray filled with zeros of the given shape. """

        return NDArray._without_copy(self._backend.zeros(shape, dtype=dtype), self)

    def ones(self, shape: Tuple[int] | List[int], dtype: str = None) -> NDArray:
        """ Return NDArray filled with ones of the given shape. """

        return NDArray._without_copy(self._backend.ones(shape, dtype=dtype), self)

    def one_hot(self, i: NDArray, num_classes: int = 10, dtype: str = None):
        assert len(i.shape) == 1, 'Expected one dimensional NDArray, i'

        # If index array device is incompatible with the used device
        if self != i.device:
            i = i.to(self)

        return NDArray._without_copy(
            self._backend.eye(num_classes, dtype=dtype)[i._data],
            self
        )

    def empty(self, shape: Tuple[int] | List[int], dtype: str = None) -> NDArray:
        """ Return uninitialized NDArray of the given shape. """

        return NDArray._without_copy(self._backend.empty(shape, dtype=dtype), self)

    def full(
        self, shape: Tuple[int] | List[int],
        fill: any = 0.,
        dtype: str = None
    ) -> NDArray:
        """ Return an NDArray of the given shape filled with provided value. """

        return NDArray._without_copy(
            self._backend.full(shape, fill, dtype=dtype),
            self
        )


## DEFAULT DEVICE ##
_DEFAULT_DEVICE = Device(DeviceType.CPU, -1)

def default_device() -> Device:
    return _DEFAULT_DEVICE

# Devices supported by Soket

def cpu() -> Device:
    return Device(DeviceType.CPU, -1)

def gpu(id: int = 0) -> Device:
    return Device(DeviceType.GPU, id)
