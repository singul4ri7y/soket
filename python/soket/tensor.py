from __future__ import annotations
from typing import Optional, List
from soket.autodiff import Node, compute_gradient
from soket.ops import *
from soket.backend.numpy import NDArray, Device, cpu, array_api
import soket
import math
import numpy

# TODO: Handle tensor counts

# If LAZY_MODE is disabled, Tensor data will be evaluated right after the operation
LAZY_MODE = False

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
        dtype: str = None,
        requires_grad: bool = None,
        **kwargs
    ):
        if isinstance(array, Tensor):
            if device is None:
                device = array.device
            if dtype is None:
                dtype = array.dtype
            # If given tensor device and dtype is same, grab the given tensor data.
            if device is array.device and dtype is array.dtype:
                cached_data = array.compute_cached_data()
            else:
                # Fall back, copy through conversion
                cached_data = Tensor._array_from_numpy(
                    array.numpy(), device=device, dtype=dtype
                )

        else:
            device = device if device else cpu()
            dtype = dtype if dtype else 'float32'
            cached_data = Tensor._array_from_numpy(array, device=device,
                dtype=dtype)

            self.init(
                None,
                [],
                cached_data=cached_data,
                requires_grad=requires_grad
            )

    @property
    def shape(self) -> Tuple:
        return self.compute_cached_data().shape

    @property
    def dtype(self) -> str:
        return self.compute_cached_data().dtype

    @property
    def device(self) -> Device:
        data = self.compute_cached_data()
        if array_api is numpy:
            return cpu()
        return data.device

    @property
    def data(self) -> Tensor:
        """ Get detached Tensor using property. """
        return self.detach()

    @data.setter
    def data(self, value):
        """ Set the value of the current Tensor without changing the variable. """

        assert isinstance(value, Tensor)
        assert self.dtype == value.dtype, 'Data-type mismatch: %s %s' %    \
            (self.dtype, value.dtype)

        self.cached_data = value.compute_cached_data()

    @property
    def t(self) -> Tensor:
        """ Property to get transposed (last two dimensions swapped) tensor """
        return self.transpose()

    @property
    def size(self) -> int:
        return self.compute_cached_data().size

    @staticmethod
    def _array_from_numpy(numpy_array, device, dtype) -> NDArray:
        if array_api is numpy:
            return numpy.array(numpy_array, dtype=dtype)
        else:
            return array_api.array(numpy_array, device=device, dtype=dtype)

    @staticmethod
    def make_const(data: Tensor | NDArray, requires_grad: bool = False):
        """ Creates a new cosntant tensor sharing the data which is detached
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
                if not isinstance(data, Tensor)
                else data.compute_cached_data(),
            requires_grad=requires_grad
        )
        return tensor

    @staticmethod
    def make_from_op(op: Op, inputs: List[Tensor]):
        tensor = Tensor.__new__(Tensor)
        tensor.init(op, inputs)

        if not LAZY_MODE:
            # Calling `tensor.init()` builds the computational graph. If output
            # tensor gradient is not required, the computational graph is not needed.
            if not tensor.requires_grad:
                return tensor.detach()
            tensor.compute_cached_data()
        return tensor

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

    def numpy(self):
        data = self.compute_cached_data()
        if array_api is numpy:
            return data
        return data.numpy()

    def transpose(self) -> Tensor:
        assert len(self.shape) >= 2, 'Tensor shape must be >= 2'
        return Transpose()(self)

    def broadcast_to(self, *shape):
        assert len(shape) != 0, 'Expected a shape'
        if isinstance(shape[0], tuple):
            shape = shape[0]

        # Only integer shape allowed
        for i in shape:
            assert isinstance(i, int), 'Shape index has to be integer'

        # Check whether the axes/dimension are compatible?
        assert len(shape) >= len(self.shape), 'Attempt to contract a tensor through broadcasting'
        rv_self_shape = reversed(self.shape)
        rv_bcast_shape = reversed(shape)

        for self_idx, bcast_idx in zip(rv_self_shape, rv_bcast_shape):
            if self_idx != 1 and self_idx != bcast_idx:
                raise ValueError(f'Shape {self.shape} and {shape} are incompatible')

        return BroadcastTo(shape)(self)

    def sum(self, *axes, keepdims: bool = False) -> Tensor:
        if len(axes) > 0 and isinstance(axes[0], tuple):
            axes = axes[0]

        # Check for dimension index under/overshoot.
        dimsiz = len(self.shape)
        for i in axes:
            assert isinstance(i, int), 'Shape index has to be integer'
            if i < -dimsiz or i >= dimsiz:
                raise ValueError(f'Axis out of bounds: {i} in {axes}')

        return Summation(None if len(axes) == 0 else axes, keepdims)(self)

    def reshape(self, *shape) -> Tensor:
        assert math.prod(shape) == math.prod(self.shape), 'Attempt to reshape with different number of element count'
        return Reshape(shape)(self)

    def permute(self, *axes) -> Tensor:
        assert len(shape) != 0, 'Expected a shape'
        if isinstance(shape[0], tuple):
            shape = shape[0]

        # Check for dimension index under/overshoot.
        dimsiz = len(self.shape)
        for i in axes:
            assert isinstance(i, int), 'Shape index has to be integer'
            if i < -dimsiz or i >= dimsiz:
                raise ValueError(f'Axis out of bounds: {i} in {axes}')

        return Permute(axes)

    def exp(self) -> Tensor:
        """ Returns Euler's exponential of this tensor. """
        return Exp()(self)

    def log(self) -> Tensor:
        """ Returns natural logarithm of this function """
        return Log()(self)

    def backward(self, grad: Optional[Tensor] = None) -> Tensor:
        """ Computes gradients of all the Nodes upto this node using Reverse Mode AD.

        Parameters
        ----------
        grad: Tensor
            Initial gradient of this node. By default it will be filled with all ones
            becasue d(out)/d(out) = 1

        """
        # Check whether gradient should be computed. If not, return.
        if not self.requires_grad:
            return

        compute_gradient(self, grad if grad else soket.ones_like(self))

    def force_grad(self):
        """ Forces the tensor to store the computed gradient in tensor.grad
        field even if it's not leaf node.
        """

        self._force_grad = True

    def __repr__(self):
        brand = 'soket.Tensor('
        prefix = ' ' * len(brand)
        lines = str(self.compute_cached_data()).splitlines()
        res = brand + '\n'.join(lines[:1] + [prefix + line for line in    \
            lines[1:]]) + f', dtype={self.dtype}'

        if self.requires_grad is True:
            res += ', requires_grad=True'

        return res + ')'

    def __add__(self, other: Tensor | any) -> Tensor:
        if isinstance(other, Tensor):
            return EWiseAdd()(self, other)
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

    def __rsub__(self, other) -> Tensor:
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
        """ This function will only be called in situations when scalar is raised to a power """
        tensor = soket.ones_like(self) * pow
        return tensor ** self

    __radd__ = __add__
    __rmul__ = __mul__
