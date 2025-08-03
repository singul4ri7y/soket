from __future__ import annotations
from typing import Optional, List
from soket.autodiff import Node, compute_gradient
from soket.ops import *
from soket.backend import NDArray, Device, DeviceType
from soket.backend.device import default_device
from soket.dtype import DType, default_datatype
import math


# If LAZY_MODE is disabled, Tensor data will be evaluated right after the operation
LAZY_MODE = False


# Decorator to adjust shapes and axes
def tensor_adjust_axes_decor(func) -> function:
    def wrapper(self, *shape, **kwargs):
        if len(shape) > 0 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])

        return func(self, shape, **kwargs)

    return wrapper

# Decorator for tensor reduction operations
def tensor_reduction_decor(func) -> function:
    @tensor_adjust_axes_decor
    def wrapper(
        self,
        axes: Tuple[int],
        dtype: DType = None,
        keepdims: bool = True
    ):
        dtype = str(DType(dtype)) if dtype else None

        # Check for dimension index under/overshoot.
        dimsiz = len(self.shape)
        for i, axis in enumerate(axes):
            assert isinstance(axis, int), f'Invalid axis: {axes}[{i}] = {axis}'
            if axis < -dimsiz or axis >= dimsiz:
                raise ValueError(f'Axis out of bounds: {axes}[{i}] = {axis}')

        return func(self, axes, dtype, keepdims)

    return wrapper


class Tensor(Node):
    """ Represents a soket Tensor """

    # Stores the gradient/adjoint. May used to hold list of partial adjoints
    # during gradient computation.
    # But it is ensured, after gradient computation this field will be a Tensor
    # with gradient value.
    grad: Tensor | List[Tensor] = None

    # Store the computed gradient even if the Node/Tensor is not leaf?
    _force_grad: bool = False

    def __init__(
        self,
        array,
        *,
        device: Optional[Device] = None,
        dtype: Optional[DType] = None,
        requires_grad: bool = False
    ) -> None:
        assert isinstance(array, (list, tuple, int, float, bool, NDArray, Tensor)),  \
            f'Invalid input to tensor {array}'

        dtype = str(DType(dtype)) if dtype else None

        if isinstance(array, Tensor):
            device = array.device if device is None else device
            cached_data = NDArray(array.compute_cached_data(), dtype=dtype,
                device=device)
        else:
            if isinstance(array, NDArray):
                device = array.device if not device else device
            else:
                device = default_device() if not device else device

                # Default datatype for tensors are float32.
                dtype = str(default_datatype) if not dtype else dtype

            cached_data = NDArray(array, device=device, dtype=dtype)

        self.init(
            None,
            [],
            cached_data=cached_data,
            requires_grad=requires_grad
        )

    ## PROPERTIES ##

    @property
    def shape(self) -> Tuple:
        """ Returns shape of the tensor. """

        return self.compute_cached_data().shape

    @property
    def dtype(self) -> DType:
        """ Get tensor datatype. """

        return DType(str(self.compute_cached_data().dtype))

    @property
    def device(self) -> Device:
        """ Returns the device tensor is using. """

        return self.compute_cached_data().device

    @property
    def data(self) -> Tensor:
        """ Get a detached tensor. """

        return self.detach()

    @data.setter
    def data(self, value: Tensor) -> None:
        """ Set the value of the current Tensor without changing the variable. """

        assert isinstance(value, Tensor), 'Expected a tensor value'
        assert self.dtype == value.dtype, 'Data-type mismatch: %s %s' %    \
            (self.dtype, value.dtype)

        self.cached_data = value.compute_cached_data()

    @property
    def T(self) -> Tensor:
        """ Returns a transposed (last two dimensions swapped) tensor. """

        return self.transpose()

    @property
    def size(self) -> int:
        """ Returns size of the tensor. """

        return self.compute_cached_data().size

    ## STATIC METHODS ##

    @staticmethod
    def make_const(data: Tensor | NDArray, requires_grad: bool = False) -> Tensor:
        """
        Creates a new cosntant tensor sharing the data which is detached
        from the computational graph.

        Parameters
        ----------
        data: Tensor or NDArray
            Newly created tensor data

        requires_grad: bool
            Should gradient be calculated for the newly created tensor if the
            tensor ends up creating a new computational graph.
        """

        # Does not call the `__init__` function
        tensor = Tensor.__new__(Tensor)
        tensor.init(
            None,
            [],
            cached_data=data
                if isinstance(data, NDArray)
                else data.compute_cached_data(),
            requires_grad=requires_grad
        )
        return tensor

    @staticmethod
    def from_numpy(array: numpy.ndarray) -> Tensor:
        """ Creates and returns a tensor from a numpy array. """

        return Tensor(NDArray(array))

    @staticmethod
    def _make_from_op(op: Op, inputs: List[Tensor]) -> Tensor:
        """ Creates a new tensor making it part of the computational graph. """

        tensor = Tensor.__new__(Tensor)
        tensor.init(op, inputs)

        if not LAZY_MODE:
            # Calling `tensor.init()` builds the computational graph. If output
            # tensor gradient is not required, the computational graph is not needed.
            if not tensor.requires_grad:
                return tensor.detach()
            tensor.compute_cached_data()

        return tensor

    ## METHODS ##

    def detach(self) -> Tensor:
        """ Creates a tensor which detaches from the graph. But using this
        tensor will result a new computational graph and original tensor data
        might get overwritten.

        Returns
        -------
        output: Tensor
            New tensor with no inputs and op (detached state from the graph)
            sharing the same data

        """

        return Tensor.make_const(self)

    def numpy(self) -> numpy.ndarray:
        """ Returns the numpy representation of tensor. """

        return self.compute_cached_data().numpy()

    def migrate_to(self, device: Device, use: Optional[bool] = True) -> Tensor:
        """ Migrates a tensor to given device and returns it. """

        return Tensor(self.compute_cached_data().migrate_to(device, use))

    def backward(self, grad: Optional[Tensor] = None) -> Tensor:
        """ Computes gradients of all the Nodes upto this node using Reverse Mode AD.

        Parameters
        ----------
        grad: Tensor
            Initial gradient of this node. By default it will be filled with all ones
            becasue d(out)/d(out) = 1

        """

        from .creation import ones_like

        # Check whether gradient should be computed. If not, return.
        if not self.requires_grad:
            return

        compute_gradient(self, grad if grad else ones_like(self))

    ## OPERATIONS ##

    @tensor_adjust_axes_decor
    def broadcast_to(self, shape) -> Tensor:
        """ Returns a tensor broadcasted to given shape. """

        assert len(shape) != 0, 'Expected a shape'

        # Only integer shape allowed
        for i in shape:
            assert isinstance(i, int), 'Shape index has to be integer'

        # Check whether the axes/dimensions are compatible?
        assert len(shape) >= len(self.shape), 'Attempt to contract a tensor through broadcasting'
        rv_self_shape = reversed(self.shape)
        rv_bcast_shape = reversed(shape)

        for self_idx, bcast_idx in zip(rv_self_shape, rv_bcast_shape):
            if self_idx != 1 and self_idx != bcast_idx:
                raise ValueError(f'Shape {self.shape} and {tuple(shape)} are incompatible')

        return BroadcastTo(shape)(self)

    @tensor_reduction_decor
    def sum(self, axes, dtype: DType = None, keepdims: bool = False) -> Tensor:
        """ Returns a new tensor performing summation reduction over given axes. """

        return Summation(None if len(axes) == 0 else axes,
            dtype=dtype, keepdims=keepdims)(self)

    @tensor_reduction_decor
    def mean(self, axes, dtype: DType = None, keepdims: bool = False) -> Tensor:
        """ Returns a new tensor performing mean reduction over given axes. """

        observations = 1
        if len(axes) == 0:
            observations = self.size
        else:
            for i in axes:
                observations *= self.shape[i]

        return Mean(observations, None if len(axes) == 0 else axes,
            dtype=dtype, keepdims=keepdims)(self)

    @tensor_adjust_axes_decor
    def reshape(self, shape: int | Tuple[int]) -> Tensor:
        assert isinstance(shape, (int, tuple)), f'Invalid shape {shape}'
        assert math.prod(shape) == self.size, 'Attempt to reshape to different number of elements'
        return Reshape(shape)(self)

    @tensor_adjust_axes_decor
    def permute(self, axes: int | Tuple[int]) -> Tensor:
        assert len(axes) != 0, 'Expected a shape'

        # Check for dimension index under/overshoot.
        dimsiz = len(self.shape)
        for i in axes:
            assert isinstance(i, int), 'Shape index has to be integer'
            if i < -dimsiz or i >= dimsiz:
                raise ValueError(f'Axis out of bounds: {i} in {axes}')

        return Permute(axes)(self)

    def transpose(self) -> Tensor:
        assert len(self.shape) >= 2, 'Tensor shape must be >= 2'
        return Transpose()(self)

    def argmax(self, axis=None, keepdims=False):
        assert isinstance(axis, int), 'Expected axis to be an integer'

        return Tensor.make_const(self.compute_cached_data().
            argmax(axis=axis, keepdims=keepdims))

    def argmin(self, axis=None, keepdims=False):
        assert isinstance(axis, int), 'Expected axis to be an integer'

        return Tensor.make_const(self.compute_cached_data().
            argmin(axis=axis, keepdims=keepdims))

    def item(self):
        """ Gets the scalar value of a scalar Tensor. """

        assert self.size == 1, 'Tensor must hold only one value to call item()'

        return self.compute_cached_data().item()

    def force_grad(self):
        """ Forces the tensor to store the computed gradient in tensor.grad
        field even if it's not leaf node.
        """

        self._force_grad = True


    ## DUNDER METHODS ##

    def __repr__(self):
        brand = 'soket.Tensor('
        prefix = ' ' * len(brand)
        lines = str(self.compute_cached_data()).splitlines()
        res = brand + '\n'.join(lines[:1] + [prefix + line for line in    \
            lines[1:]]) + f', dtype={self.dtype}'

        if self.device.type == DeviceType.GPU:
            res += f', device=GPU:{self.device.id}'

        if self.requires_grad is True:
            res += ', requires_grad=True'

        return res + ')'

    ## DUNDER OPERATIONS ##

    def __add__(self, other: Tensor | any) -> Tensor:
        """ Adds and returns two Tensors (also supports scalar addtion). """

        # Check if devices are compatible
        assert self.device == other.device, f'Incompatible devices {self.device} and {other.device}'

        if isinstance(other, Tensor):
            return EWiseAdd()(self, other)

        assert isinstance(other, (int, float, bool)), f'Invalid appends {other}'
        return AddScalar(other)(self)

    def __sub__(self, other: Tensor | any) -> Tensor:
        if isinstance(other, Tensor):
            return EWiseAdd()(self, -other)
        return AddScalar(-other)(self)

    def __mul__(self, other: Tensor | any) -> Tensor:
        if isinstance(other, Tensor):
            return EWiseMul()(self, other)
        return MulScalar(other)(self)

    def __truediv__(self, other: Tensor | any) -> Tensor:
        if isinstance(other, Tensor):
            return EWiseDiv()(self, other)
        return DivScalar(other)(self)

    def __neg__(self) -> Tensor:
        return Negate()(self)

    def __rsub__(self, other: Tensor | any) -> Tensor:
        if isinstance(other, Tensor):
            return EWiseAdd()(-self, other)
        return AddScalar(other)(-self)

    def __matmul__(self, other: Tensor) -> Tensor:
        assert len(self.shape) >= 2, f'Tensor should atleast have 2 dimensions, shape: {self.shape}'
        assert len(other.shape) >= 2, f'Tensor should atleast have 2 dimensions, shape: {other.shape}'
        assert self.shape[-1] == other.shape[-2], 'Tensor dimensions should be compatible for matmul'

        return MatMul()(self, other)

    def __pow__(self, pow: Tensor | any) -> Tensor:
        if isinstance(pow, Tensor):
            return EWisePow()(self, pow)
        return PowerScalar(pow)(self)

    def __rpow__(self, pow: any) -> Tensor:
        """ This function will only be called in situations when scalar is
        raised to a power
        """

        from .creation import full

        tensor = full(self.shape, c=pow)
        return EWisePow()(tensor, self)

    __radd__ = __add__
    __rmul__ = __mul__

    ## Methods below are not part of computational grpah ##
    ## Hence, there is no lazy evaluation because no input tensor is stored ##

    def __gt__(self, other) -> Tensor:
        """ Greater than """
        cached_data = self.compute_cached_data() > other.compute_cached_data() \
            if isinstance(other, Tensor) else self.compute_cached_data() > other

        return Tensor.make_const(cached_data, requires_grad=False)

    def __ge__(self, other) -> Tensor:
        """ Greater than and equal """
        cached_data = self.compute_cached_data() >= other.compute_cached_data() \
            if isinstance(other, Tensor) else self.compute_cached_data() >= other

        return Tensor.make_const(cached_data, requires_grad=False)

    def __rgt__(self, other) -> Tensor:
        """ Greater than (RHS)"""
        cached_data = other.compute_cached_data() > self.compute_cached_data() \
            if isinstance(other, Tensor) else other > self.compute_cached_data()

        return Tensor.make_const(cached_data, requires_grad=False)

    def __rge__(self, other) -> Tensor:
        """ Greater than and equal (RHS) """
        cached_data = other.compute_cached_data() >= self.compute_cached_data() \
            if isinstance(other, Tensor) else other >= self.compute_cached_data()

        return Tensor.make_const(cached_data, requires_grad=False)

    def __lt__(self, other) -> Tensor:
        """ Less than """
        cached_data = self.compute_cached_data() < other.compute_cached_data() \
            if isinstance(other, Tensor) else self.compute_cached_data() < other

        return Tensor.make_const(cached_data, requires_grad=False)

    def __le__(self, other) -> Tensor:
        """ Less than and equal """
        cached_data = self.compute_cached_data() <= other.compute_cached_data() \
            if isinstance(other, Tensor) else self.compute_cached_data() <= other

        return Tensor.make_const(cached_data, requires_grad=False)

    def __rlt__(self, other) -> Tensor:
        """ Less than (RHS)"""
        cached_data = other.compute_cached_data() < self.compute_cached_data() \
            if isinstance(other, Tensor) else other < self.compute_cached_data()

        return Tensor.make_const(cached_data, requires_grad=False)

    def __rle__(self, other) -> Tensor:
        """ Less than and equal (RHS) """
        cached_data = other.compute_cached_data() <= self.compute_cached_data() \
            if isinstance(other, Tensor) else other <= self.compute_cached_data()

        return Tensor.make_const(cached_data, requires_grad=False)

    def __eq__(self, other: Tensor) -> Tensor:
        return Tensor.make_const(self.compute_cached_data() ==
            other.compute_cached_data())

    def __ne__(self, other: Tensor) -> Tensor:
        return Tensor.make_const(self.compute_cached_data() !=
            other.compute_cached_data())

    def __getitem__(self, idx: int | slice | Tuple[int | slice]):
        return Select(idx)(self)

    def __setitem__(
        self,
        idx: int | slice | Tuple[int | slice],
        value
    ):
        assert self.requires_grad is False, 'Cannot set value to a parameter'

        if isinstance(value, Tensor):
            value = value.compute_cached_data()
        elif isinstance(value, numpy):
            value = value.tolist()

        return Tensor.make_const(self.compute_cached_data().__setitem__(
            idx, value
        ))
