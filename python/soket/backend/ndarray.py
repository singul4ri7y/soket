from __future__ import annotations
from typing import List, Tuple, Optional
import math
import numpy
import cupy


## NOTE: Writing datatypes as strings is recommended for backend code instead of using soket.dtype.

# NDArray reduction operation decorator for verifying valid axis.
def ndarray_reduction_decor(func) -> function:
    def wrapper(
        self,
        axes: Optional[int | Tuple[int] | List[int]] = None,
        dtype: str = None,
        keepdims: bool = False
    ) -> NDArray:
        if axes is not None:
            assert isinstance(axes, (int, tuple, list)), f'Invalid axes {axes}'

            # Check for dimension index under/overshoot.
            dimsiz = len(self.shape)
            if isinstance(axes, int):
                if axes < -dimsiz or axes >= dimsiz:
                    raise ValueError(f'Axis {axes} is out of bounds')
            for i, axis in enumerate(axes):
                assert isinstance(axis, int), f'Invalid axis: {axes}[{i}] = {axis}'
                if axis < -dimsiz or axis >= dimsiz:
                    raise ValueError(f'Axis out of bounds: {axes}[{i}] = {axis}')

        return NDArray._without_copy(func(self, axes, dtype, keepdims), self.device)

    return wrapper

# NDArray static method decorator for checking NDArray specific input
def ndarray_static_decor_unary(func) -> function:
    def wrapper(x, *args, **kwargs) -> NDArray:
        assert isinstance(x, NDArray), 'Expected an NDArray as input'
        return NDArray._without_copy(func(x, *args, **kwargs), device=x.device)

    return wrapper

# NDArray element-wise method decorator for checking input validity and ensuring
# consistent type promotion for different dtypes.
def ndarray_elemwise_decor(func) -> function:
    def wrapper(self, other: NDArray) -> NDArray:
        assert isinstance(other, (int, float, bool, NDArray)), f'Invalid RHS input {other}'



class NDArray:
    """
    NDArray - the core backend for multidimensional array manipulation for Soket.
    NDArray can use CPU or GPU operations depending on the soket.Device it is
    created with.
    """

    _data: numpy.ndarray | cupy.ndarray
    _device: object  # Device the array is created with

    def __init__(
        self,
        array: any,
        device: object = None,
        dtype: str = None
    ):
        from .device import default_device

        assert isinstance(array, (list, tuple, int, float, bool, numpy.ndarray,
            cupy.ndarray, NDArray)), f'Invalid input {array}'

        if isinstance(array, NDArray):
            device = device if device else array.device

            if device == array.device:
                self._data = device._backend.array(array._data, dtype=dtype)
            else:
                # Move (if need be) to provided device and copy the data.
                self._data = device._backend.array(array.migrate_to(device)._data, dtype=dtype)

        # Raw NumPy (CPU) or CuPy (GPU) array
        elif isinstance(array, (numpy.ndarray, cupy.ndarray)):
            from . import DeviceType

            device = default_device() if not device else device

            # CuPy (GPU) supports NumPy arrays
            if isinstance(array, device._backend.ndarray) or device.type == DeviceType.GPU:
                self._data = device._backend.array(array, dtype=dtype)
            else:
                # If using NumPy (CPU) backend but working with CuPy (GPU) array
                self._data = device._backend.array(cupy.asnumpy(array), dtype=dtype)
        else:
            device = default_device() if not device else device
            self._data = device._backend.array(array, dtype=dtype)

        # Store the device we are creating the array from.
        self._device = device

    ## PROPERTIES ##

    @property
    def shape(self) -> tuple:
        """ Returns shape of the n-d array. """

        return self._data.shape

    @property
    def dtype(self) -> str:
        """ Returns the datatype used in n-d array. """

        return str(self._data.dtype)

    @property
    def device(self) -> object:
        """ Returns the device array is using. """

        return self._device

    @property
    def size(self) -> int:
        """ Returns the size of the n-d array. """

        return self._data.size

    ## METHODS ##

    def migrate_to(self, device: object, use: bool = True) -> NDArray:
        """ Migrate NDArray to a new device. """

        if self.device == device:
            return NDArray(self._data, device=self.device)

        # Use the device (for GPU only)
        if use is True:
            device.use()

        return NDArray(self._data, device=device)

    def numpy(self) -> numpy.ndarray:
        """ Returns numpy.ndarray representation of array. """

        from . import DeviceType

        # CPU backend is NumPy.
        if self.device.type == DeviceType.CPU:
            return self._data

        # For GPU backend
        return self.device._backend.asnumpy(self._data)

    ## OPERATIONS ##

    @ndarray_reduction_decor
    def sum(
        self,
        axes: Optional[int | Tuple[int] | List[int]] = None,
        dtype: str = None,
        keepdims: Optional[bool] = False
    ):
        """ Returns array after performing summation reduction. """

        res = self._data.sum(axes, dtype=dtype, keepdims=keepdims)
        self.device.sync()

        # NumPy can return scalar
        if not isinstance(res, self.device._backend.ndarray):
            res = self.device._backend.array(res)

        return res

    @ndarray_reduction_decor
    def mean(
        self,
        axes: Optional[int | Tuple[int]] = None,
        dtype: str = None,
        keepdims: Optional[bool] = False
    ):
        """ Returns array after performing mean reduction. """

        res = self._data.mean(axes, dtype=dtype, keepdims=keepdims)
        self.device.sync()

        # NumPy can return scalar
        if not isinstance(res, self.device._backend.ndarray):
            res = self.device._backend.array(res)

        return res

    def reshape(
        self,
        shape: int | Tuple[int]
    ) -> NDArray:
        """ Returns the reshaped n-d array of current n-d array. """

        assert isinstance(shape, (int, tuple)), f'Invalid shape {shape}'
        for i, s in enumerate(shape):
            assert isinstance(i, int), f'Expected shape to be an integer: {shape}[{i}] = {s}'
        assert math.prod(shape) == self.size, 'Attempt to reshape to different number of elements'

        return NDArray._without_copy(self._data.reshape(shape), self.device)

    def argmax(
        self,
        axis: Optional[int] = None,
        keepdims: Optional[bool] = False
    ) -> NDArray:
        """ Returns the indicies NDArray of the maximum values along an axis. """

        assert isinstance(axis, int), 'Expected axis to be an integer'

        return self._data.argmax(axis, keepdims=keepdims)

    def argmin(
        self,
        axis: Optional[int] = None,
        keepdims: Optional[bool] = False
    ) -> NDArray:
        """ Returns the indicies NDArray of the minimum values along an axis. """

        assert isinstance(axis, int), 'Expected axis to be an integer'

        return self._data.argmin(axis, keepdims=keepdims)

    ## DUNDER METHODS ##

    def __str__(self) -> str:
        return self._data.__str__()

    ## DUNDER OPERATIONS ##

    def __add__(self, other: NDArray) -> NDArray:
        """ Adds and return two array, also supports scalar addition. """





    ## STATIC METHODS ##

    @staticmethod
    def _without_copy(x: numpy.ndarray | cupy.ndarray, device: object) -> NDArray:
        """ THIS METHOD TO BE USED ONLY BY THE NDArray OPERATOR FUNCTIONS. """
        """
        This method only creates an NDArray instance filling in the data w/o
        doing any sanity check.
        """

        assert isinstance(x, (device._backend.ndarray)), f'Device incompatible array provided'

        ndarray = NDArray.__new__(NDArray)
        ndarray._data = x
        ndarray._device = device

        return ndarray

    ## STATIC OPERATIONS ##

    @staticmethod
    @ndarray_static_decor_unary
    def log(x: NDArray, dtype: str = None):
        """ Performs element-wise natural log. """

        res = x.device._backend.log(x._data, dtype=dtype)
        x.device.sync()  # Sync current thread with GPU computation (if need be)

        return res

    @staticmethod
    @ndarray_static_decor_unary
    def exp(x: NDArray, dtype: str = None):
        """ Performs element-wise Euler's exponential. """

        res = x.device._backend.exp(x._data, dtype=dtype)
        x.device.sync()  # Sync current thread with GPU computation (if need be)

        return res

    @staticmethod
    @ndarray_static_decor_unary
    def permute(x: NDArray, axes: Optional[int | Tuple[int]] = None):
        """ Returns an array with axes permuted. """

        if axes is not None:
            assert isinstance(axes, (int, tuple, list)), f'Invalid axes {axes}'

            if isinstance(axes, (list, tuple)):
                assert len(axes) == len(x.shape), f'Permute axes should cover all axis of array'
                for i, axis in enumerate(axes):
                    assert isinstance(axis, int), f'Invalid axis {axis} on index {i}'

        res = x.device._backend.transpose(x._data, axes)
        x.device.sync()  # Sync current thread with GPU computation (if need be)

        return res

    @staticmethod
    @ndarray_static_decor_unary
    def swapaxes(
        x: NDArray,
        axis1: int,
        axis2: int
    ) -> NDArray:
        assert isinstance(axis1, int), f'Invalid axis1'
        assert isinstance(axis2, int), f'Invalid axis2'

        return NDArray(x._data.swapaxes(axis1, axis2))

    @staticmethod
    def maximum(x, y, dtype: str = None) -> NDArray:
        """ Returns the element-wise maximum of two arrays. """

        from soket import default_device

        assert isinstance(x, (int, float, bool, list, tuple, NDArray)), f'Invalid input {x}'
        assert isinstance(y, (int, float, bool, list, tuple, NDArray)), f'Invalid input {y}'

        x_ndarray_instance = isinstance(x, NDArray)
        y_ndarray_instance = isinstance(y, NDArray)

        if x_ndarray_instance and y_ndarray_instance:
            assert x.device == y.device, f'Incompatible devices {x.device} and {y.device}'

            device = x.device
        elif x_ndarray_instance or y_ndarray_instance:
            device = x.device if x_ndarray_instance else y.device

            x = NDArray(x, device=device, dtype=dtype) if not x_ndarray_instance else x
            y = NDArray(y, device=device, dtype=dtype) if not y_ndarray_instance else y
        else:
            device = default_device()

            x = NDArray(x, device=device, dtype=dtype)
            y = NDArray(y, device=device, dtype=dtype)

        res = device._backend.maximum(x._data, y._data, dtype=dtype)
        device.sync()  # Sync current thread with GPU computation (if need be)

        return NDArray(res, device=device)

    @staticmethod
    @ndarray_static_decor_unary
    def broadcast_to(x: NDArray, shape: Tuple[int]):
        """ Broadcast an n-d array to given shape. """

        assert isinstance(shape, tuple), f'Invalid shape {shape}'
        for i, s in enumerate(shape):
            assert isinstance(s, int), f'Invalid shape {s} on index {i}'

        res = x.device._backend.broadcast_to(x._data, shape)
        x.device.sync()  # Sync current thread with GPU computation (if need be)

        return res
