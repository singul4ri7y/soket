from __future__ import annotations
from typing import List, Tuple, Optional, Sequence
from soket.dtype import promote_types, get_scalar_dtype
import math
import numpy
import cupy


## NOTE: Writing datatypes as strings is recommended for backend code instead of using soket.dtype.


# Always ensure ndarray, especially for case of NumPy, where np.float64 (and friends)
# type output might be possible.
def ndarray_ensure(object: object, device: object) -> numpy.ndarray | cupy.ndarray:
    if isinstance(object, device._backend.ndarray):
        return object
    else:
        return device._backend.array(object)


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
                
        res = func(self, axes, dtype, keepdims)
        return NDArray._without_copy(ndarray_ensure(res, self.device), self.device)

    return wrapper

# NDArray max/min operation decorator only for validating axis.
def ndarray_max_min_decor(func) -> function:
    def wrapper(
        self,
        axes: Optional[int | Tuple[int] | List[int]] = None,
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
                
        res = func(self, axes, keepdims)
        return NDArray._without_copy(ndarray_ensure(res, self.device), self.device)

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
        if isinstance(other, NDArray):
            assert self.device == other.device, f'Incompatible array device'

            res = func(self._data, other._data)

            # Use custom datatype promotion (if need be).
            if self.dtype != other.dtype:
                dtype = str(promote_types(self.dtype, other.dtype))

                # If type promotion done differently.
                if dtype != res.dtype:
                    res = res.astype(dtype)
        elif isinstance(other, (int, float, bool)):
            scalar_ndarray = self.device._backend.array(other, dtype=self.dtype)
            res = func(self._data, scalar_ndarray)
        else:
            raise ValueError(f'Invalid value {other}')
        
        return NDArray._without_copy(ndarray_ensure(res, self.device), device=self.device)
    
    return wrapper

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

        return self._data.sum(axes, dtype=dtype, keepdims=keepdims)

    @ndarray_reduction_decor
    def mean(
        self,
        axes: Optional[int | Tuple[int]] = None,
        dtype: str = None,
        keepdims: Optional[bool] = False
    ):
        """ Returns array after performing mean reduction. """

        return self._data.mean(axes, dtype=dtype, keepdims=keepdims)
    
    @ndarray_max_min_decor
    def max(
        self,
        axes: Optional[int | Tuple[int]] = None,
        keepdims: Optional[bool] = False
    ):
        """ Finds and returns maximum numbers from an array. """

        return self._data.max(axes, keepdims=keepdims)
    
    @ndarray_max_min_decor
    def min(
        self,
        axes: Optional[int | Tuple[int]] = None,
        keepdims: Optional[bool] = False
    ):
        """ Finds and returns maximum numbers from an array. """

        return self._data.min(axes, keepdims=keepdims)

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

        return NDArray._without_copy(
            self._data.argmax(axis, keepdims=keepdims),
            self.device
        )

    def argmin(
        self,
        axis: Optional[int] = None,
        keepdims: Optional[bool] = False
    ) -> NDArray:
        """ Returns the indicies NDArray of the minimum values along an axis. """

        assert isinstance(axis, int), 'Expected axis to be an integer'

        return NDArray._without_copy(
            self._data.argmin(axis, keepdims=keepdims),
            self.device
        )
    
    def item(self) -> any:
        """ Get scalar item from arrays with only one element. """

        assert self.size == 1, f'Array must have only one element.'
        return self._data.item()

    ## DUNDER METHODS ##

    def __str__(self) -> str:
        return self._data.__str__()

    ## DUNDER OPERATIONS ##
    
    @ndarray_elemwise_decor
    def __add__(
        self_data: numpy.ndarray | cupy.ndarray,
        other: numpy.ndarray | cupy.ndarray
    ):
        """ Performs addition on two arrays. Supports scalars. """

        return self_data + other
    
    __radd__ = __add__
    
    def __neg__(self) -> NDArray:
        """ Returns a negated array. """
    
        return NDArray._without_copy(-self._data, device=self.device)
    
    @ndarray_elemwise_decor
    def __sub__(
        self_data: numpy.ndarray | cupy.ndarray,
        other: numpy.ndarray | cupy.ndarray
    ):
        """
        Performs subtraction between two array and returns it.
        Supports scalars.
        """

        # Operator function from LHS
        return self_data - other
    
    @ndarray_elemwise_decor
    def __rsub__(
        self_data: numpy.ndarray | cupy.ndarray,
        other: numpy.ndarray | cupy.ndarray
    ):
        """
        Performs subtraction between two array and returns it.
        Supports scalars.
        """

        # Operator function from RHS
        return other - self_data
    
    @ndarray_elemwise_decor
    def __mul__(
        self_data: numpy.ndarray | cupy.ndarray,
        other: numpy.ndarray | cupy.ndarray
    ):
        """ Performs multiplication on two arrays. Supports scalars. """

        return self_data * other
    
    __rmul__ = __mul__
    
    @ndarray_elemwise_decor
    def __truediv__(
        self_data: numpy.ndarray | cupy.ndarray,
        other: numpy.ndarray | cupy.ndarray
    ):
        """
        Divides an array with another array (or scalar). Returns the result.
        """

        # Operator function from LHS

        return self_data / other

    @ndarray_elemwise_decor
    def __rtruediv__(
        self_data: numpy.ndarray | cupy.ndarray,
        other: numpy.ndarray | cupy.ndarray
    ):
        """
        Divides an array with another array (or scalar). Returns the result.
        """

        # Operator function from RHS

        return other / self_data
    
    @ndarray_elemwise_decor
    def __pow__(
        self_data: numpy.ndarray | cupy.ndarray,
        other: numpy.ndarray | cupy.ndarray
    ):
        """ Performs element-wise power operation on array. Supports scalar. """

        # Operator function from LHS

        return self_data ** other
    
    @ndarray_elemwise_decor
    def __rpow__(
        self_data: numpy.ndarray | cupy.ndarray,
        other: numpy.ndarray | cupy.ndarray
    ):
        """ Performs element-wise power operation on array. Supports scalar. """

        # Operator function from RHS

        return other ** self_data

    def __matmul__(self, other: NDArray) -> NDArray:
        """ Perform matrix-matrix multiplication of arrays. """

        assert isinstance(other, NDArray), f'Matrix/array must be NDArray'
        assert self.device == other.device, f'Attempt to matmul NDArray from different '
        f'devices, {self.device} and {other.device}'
        
        assert len(self.shape) >= 2, f'NDArray should atleast have 2 dimensions, shape {self.shape}'
        assert len(other.shape) >= 2, f'NDArray should atleast have 2 dimensions, shape {other.shape}'

        return NDArray._without_copy(self._data @ other._data, self.device)
    
    @ndarray_elemwise_decor
    def __gt__(
        self_data: numpy.ndarray | cupy.ndarray,
        other: numpy.ndarray | cupy.ndarray
    ):
        """ Checks whether current array is greater than an element. """

        # LHS operation

        return self_data > other
    
    @ndarray_elemwise_decor
    def __rgt__(
        self_data: numpy.ndarray | cupy.ndarray,
        other: numpy.ndarray | cupy.ndarray
    ):
        """ Checks whether current array is greater than an element. """

        # RHS operation

        return other > self_data
    
    @ndarray_elemwise_decor
    def __ge__(
        self_data: numpy.ndarray | cupy.ndarray,
        other: numpy.ndarray | cupy.ndarray
    ):
        """ Checks whether current array is greater than and equal to an element. """

        # LHS operation

        return self_data >= other
    
    @ndarray_elemwise_decor
    def __rge__(
        self_data: numpy.ndarray | cupy.ndarray,
        other: numpy.ndarray | cupy.ndarray
    ):
        """ Checks whether current array is greater than and equal to an element. """

        # RHS operation

        return other >= self_data
    
    @ndarray_elemwise_decor
    def __lt__(
        self_data: numpy.ndarray | cupy.ndarray,
        other: numpy.ndarray | cupy.ndarray
    ):
        """ Checks whether current array is less than an element. """

        # LHS operation

        return self_data < other
    
    @ndarray_elemwise_decor
    def __rlt__(
        self_data: numpy.ndarray | cupy.ndarray,
        other: numpy.ndarray | cupy.ndarray
    ):
        """ Checks whether current array is less than an element. """

        # RHS operation

        return other > self_data
    
    @ndarray_elemwise_decor
    def __le__(
        self_data: numpy.ndarray | cupy.ndarray,
        other: numpy.ndarray | cupy.ndarray
    ):
        """ Checks whether current array is less than and equal to an element. """

        # LHS operation

        return self_data <= other
    
    @ndarray_elemwise_decor
    def __rle__(
        self_data: numpy.ndarray | cupy.ndarray,
        other: numpy.ndarray | cupy.ndarray
    ):
        """ Checks whether current array is less than and equal to an element. """

        # RHS operation

        return other <= self_data

    def __eq__(self, other: NDArray) -> NDArray:
        """ Checks whether current array is equal to another array. """

        assert isinstance(other, NDArray), f'Expected an NDArray, got {other}'
        assert self.device == other.device, f'Attempt to compare arrays from different '
        f'devices, found {self.device} and {other.device}'

        res = self._data == other._data
    
        return NDArray._without_copy(
            ndarray_ensure(res, self.device),
            self.device
        )
    
    def __ne__(self, other: NDArray) -> NDArray:
        """ Checks whether current array is equal to another array. """

        assert isinstance(other, NDArray), f'Expected an NDArray, got {other}'
        assert self.device == other.device, f'Attempt to compare arrays from different '
        f'devices, found {self.device} and {other.device}'

        res = self._data != other._data

        return NDArray._without_copy(
            ndarray_ensure(res, self.device),
            self.device
        )
    
    def __getitem__(
        self,
        idx: int | slice | Tuple[int | slice]
    ) -> NDArray:
        """ Gets element(s) from given indicies. """

        if isinstance(idx, tuple):
            for i, ii in enumerate(idx):
                assert isinstance(i, (int, slice)), f'Invalid index {ii} on {idx}[{i}]'
        assert isinstance(idx, (int, slice)), f'Invalid index {idx}'

        res = self._data.__getitem__(idx)

        return NDArray._without_copy(ndarray_ensure(res, self.device), self.device)
    
    ## Does not create a new array
    def __setitem__(
        self,
        idx: int | slice | Tuple[int | slice],
        value: NDArray | any
    ) -> None:
        """ Sets element(s) to indicies of an array. """

        if isinstance(idx, tuple):
            for i, ii in enumerate(idx):
                assert isinstance(i, (int, slice)), f'Invalid index {ii} on {idx}[{i}]'
        assert isinstance(idx, (int, slice)), f'Invalid index {idx}'

        if isinstance(value, NDArray):
            value = value.migrate_to(self.device)._data
        else:
            assert isinstance(value, (int, float, bool)), f'Unexpected input, {value}'

        self._data.__setitem__(idx, value)

    ## STATIC METHODS ##

    @staticmethod
    def _without_copy(x: numpy.ndarray | cupy.ndarray, device: object) -> NDArray:
        """ THIS METHOD TO BE USED ONLY BY THE NDArray OPERATOR FUNCTIONS. """
        """
        This method only creates an NDArray instance filling in the data w/o
        doing any sanity check.
        """

        assert isinstance(x, device._backend.ndarray), f'Device incompatible array provided'

        ndarray = NDArray.__new__(NDArray)
        ndarray._data = x
        ndarray._device = device

        return ndarray

    ## STATIC OPERATIONS ##

    @staticmethod
    @ndarray_static_decor_unary
    def log(x: NDArray, dtype: str = None):
        """ Performs element-wise natural log. """

        return x.device._backend.log(x._data, dtype=dtype)

    @staticmethod
    @ndarray_static_decor_unary
    def exp(x: NDArray, dtype: str = None):
        """ Performs element-wise Euler's exponential. """

        return x.device._backend.exp(x._data, dtype=dtype)

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

        return x.device._backend.transpose(x._data, axes)

    @staticmethod
    @ndarray_static_decor_unary
    def swapaxes(
        x: NDArray,
        axis1: int,
        axis2: int
    ) -> NDArray:
        """ Swap two axis denoted by given indicies. """
    
        assert len(x.shape) >= 2, f'Array should consist atleast two dimensions'
        assert isinstance(axis1, int), f'Invalid axis1'
        assert isinstance(axis2, int), f'Invalid axis2'

        return x._data.swapaxes(axis1, axis2)

    @staticmethod
    def maximum(x, y, dtype: str = None) -> NDArray:
        """ Returns the element-wise maximum of two arrays. """

        from soket import default_device

        assert isinstance(x, (int, float, bool, NDArray)), f'Invalid input {x}'
        assert isinstance(y, (int, float, bool, NDArray)), f'Invalid input {y}'

        x_ndarray_instance = isinstance(x, NDArray)
        y_ndarray_instance = isinstance(y, NDArray)

        if x_ndarray_instance and y_ndarray_instance:
            assert x.device == y.device, f'Incompatible devices {x.device} and {y.device}'
            device = x.device

            # Check for appropriate type promotion.
            if x.dtype != y.dtype:
                dtype = str(promote_types(x.dtype, y.dtype))

            x = x._data
            y = y._data
        elif x_ndarray_instance or y_ndarray_instance:
            device = x.device if x_ndarray_instance else y.device
            dtype = x.dtype if x_ndarray_instance else y.dtype

            x = x._data if x_ndarray_instance else x
            y = y._data if y_ndarray_instance else y
        else:
            device = default_device()
            dtype = str(promote_types(
                get_scalar_dtype(x),
                get_scalar_dtype(y)
            ))

        res = device._backend.maximum(x, y, dtype=dtype)
        return NDArray(ndarray_ensure(res, device), device=device)

    @staticmethod
    @ndarray_static_decor_unary
    def broadcast_to(x: NDArray, shape: Tuple[int]):
        """ Broadcast an n-d array to given shape. """

        assert isinstance(shape, tuple), f'Invalid shape {shape}'
        for i, s in enumerate(shape):
            assert isinstance(s, int), f'Invalid shape {s} on index {i}'

        return x.device._backend.broadcast_to(x._data, shape)
    
    @staticmethod
    def stack(arrays: Sequence[NDArray], axis: int = 0, dtype: str = None):
        """
        Stacks a sequence of NDArrays. Default device is the first elements device.
        """

        for array in arrays:
            assert isinstance(array, NDArray), f'Expected a sequence of NDArrays'

        device = arrays[0].device
        return NDArray._without_copy(device._backend.stack(
            [x._data for x in arrays],
            axis=axis,
            dtype=dtype
        ), device)
