from __future__ import annotations
from typing import Tuple, Union, Optional
from soket.backend.numpy import NDArray, array_api
import numpy

class Op:
    """ Operator class, works as a interface class to tensor specific
    operator classes

    """

    def __call__(self, *args):
        raise NotImplementedError()

    def compute(self, *args: Tuple[NDArray]) -> NDArray:
        """ Calculate forward pass of the operator

        Parameters
        ----------
        input: Tuple[NDArray]
            A tuple of input arrays to the function

        Returns
        -------
        output: NDArray
            Output of the operation

        """
        raise NotImplementedError()

    def gradient(self, node: Node, adj: Node) -> Union[Node, Tuple[Node]]:
        """ Computes the partial adjoints of the inputs to current/this
        node in computational graph.

        Parameters
        ----------
        node: Node
            Currently processed value/node of the computational graph

        adj: Node
            Adjoint/gradient value of the node

        Returns
        -------
        input_gradients: Node or Tuple[Node]
            A list containing partial adjoints/gradients of inputs to this node

        """
        raise NotImplementedError()

class TensorOp:
    """ Oeperator class geared towards Tensors """

    def __call__(self, *args):
        from soket.tensor import Tensor
        return Tensor.make_from_op(self, args)


## TENSOR OPERATIONS ##

class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a + b

    def gradient(self, node: Tensor, adj: Tensor) -> Tuple[Tensor]:
        grad_a = adj if node.inputs[0].requires_grad else None
        grad_b = adj if node.inputs[1].requires_grad else None

        return grad_a, grad_b

def add(a: Tensor, b: Tensor) -> Tensor:
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar: any):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return a + self.scalar

    def gradient(self, node: Tensor, adj: Tensor) -> Tuple[Tensor]:
        if not node.inputs[0].requires_grad:
            return (None,)
        return (adj,)

def add_scalar(a: Tensor, scalar: any) -> Tensor:
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a * b

    def gradient(self, node: Tensor, adj: Tensor) -> Tuple[Tensor]:
        lhs, rhs = node.inputs
        grad_lhs = adj * rhs if lhs.requires_grad else None
        grad_rhs = adj * lhs if rhs.requires_grad else None
        return grad_lhs, grad_rhs

def multiply(a: Tensor, b: Tensor) -> Tensor:
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar: any):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return a * self.scalar

    def gradient(self, node: Tensor, adj: Tensor) -> Tuple[Tensor]:
        if not node.inputs[0].requires_grad:
            return (None,)
        return (adj * self.scalar,)

def mul_scalar(a: Tensor, scalar: any) -> Tensor:
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """ Op to raise a tensor to an (integer) power. """

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return a ** self.scalar

    def gradient(self, node: Tensor, adj: Tensor) -> Tuple[Tensor]:
        input = node.inputs[0]
        if not input.requires_grad:
            return (None,)

        return ((self.scalar * (input ** (self.scalar - 1))) * adj,)

def power_scalar(a: Tensor, scalar: int) -> Tensor:
    return PowerScalar(scalar)(a)


class EWisePow(TensorOp):
    """ Op to element-wise raise a tensor to a power. """

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a ** b

    def gradient(self, node: Tensor, adj: Tensor) -> Tuple[Tensor]:
        a, b = node.inputs

        grad_a = adj * b * (a ** (b - 1)) if a.requires_grad else None
        grad_b = adj * (a ** b) * array_api.log(a.cached_data) \
            if b.requires_grad else None

        return grad_a, grad_b

def power(a: Tensor, b: Tensor) -> Tensor:
    return EWisePow()(a, b)


class EWiseDiv(TensorOp):
    """ Op to element-wise divide two nodes. """

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a / b

    def gradient(self, node: Tensor, adj: Tensor) -> Tuple[Tensor]:
        a, b = node.inputs

        grad_a = adj / a if a.requires_grad else None
        grad_b = - adj * a / (b ** 2) if b.requires_grad else None

        return grad_a, grad_b

def divide(a: Tensor, b: Tensor) -> Tensor:
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a) -> NDArray:
        return a / self.scalar

    def gradient(self, node: Tensor, adj: Tensor) -> Tuple[Tensor]:
        if not node.inputs[0].requires_grad:
            return (None,)
        return (adj / self.scalar,)

def divide_scalar(a: Tensor, scalar: any) -> Tensor:
    return DivScalar(scalar)(a)


class Permute(TensorOp):
    """ Op to re-arrange a tensor """
    def __init__(self, axes: Tuple[int]):
        self.axes = axes

    def compute(self, a: NDArray) -> NDArray:
        if array_api is numpy:
            return numpy.transpose(a, self.axes)
        return array_api.permute(a, self.axes)

    def gradient(self, node: Tensor, adj: Tensor) -> Tuple[Tensor]:
        if not node.inputs[0].requires_grad:
            return (None,)

        # Compute the inverse axes
        if not self.inv_axes:
            self.inv_axes = [0] * len(self.axes)
            for i, x in self.axes:
                self.inv_axes[x] = i

        return (adj.permute(self.inv_axes),)

def permute(a: Tensor, axes: Optional[Tuple[int]]) -> Tensor:
    return Permute(axes)(a)


class Transpose(Permute):
    def __init__(self):
        super().__init__(None)

    """ Only re-arrange last two dimensions """
    def compute(self, a: NDArray) -> NDArray:
        if self.axes is None:
            shape_len = len(a.shape)
            self.axes = tuple(range(shape_len - 2)) + (shape_len - 1, shape_len - 2)

        return super().compute(a)

    def gradient(self, node: Tensor, adj: Tensor) -> Tensor:
        return super().gradient(node, adj)

def transpose(a: Tensor) -> Tensor:
    return Transpose()(a)


class Reshape(TensorOp):
    def __init__(self, shape: Tuple[int]):
        self.shape = shape

    def compute(self, a: NDArray) -> NDArray:
        return a.reshape(self.shape)

    def gradient(self, node: Tensor, adj: Tensor) -> Tuple[Tensor]:
        x = node.inputs[0]
        if not x.requires_grad:
            return (None,)

        return (adj.reshape(x.shape),)

def reshape(a: Tensor, shape: Tuple[int]) -> Tensor:
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape: Tuple[int]):
        self.shape = shape

    def compute(self, a: NDArray) -> NDArray:
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, node: Tensor, adj: Tensor) -> Tensor:
        x = node.inputs[0]
        if not x.requires_grad:
            return (None,)

        # Compute the summation shape if not computed
        if not self.sum_shape:
            rv_bcast_axes = reversed(self.axes)
            rv_input_axes = reversed(x.shape)
            bcast_axes_len = len(rv_bcast_axes)
            input_axes_len = len(rv_input_axes)
            diff = bcast_axes_len - input_axes_len

            sum_shape = list(range(diff))
            for i, (input_shape, bcast_shape) in \
            enumerate(zip(rv_bcast_axes, rv_input_axes)):
                # Assuming broadcast shape is compatible
                if input_shape != bcast_shape:
                    # The index is for reversed shape. Also adjust for missing
                    # dimension in the input tensor.
                    sum_shape.append(diff + (input_axes_len - i - 1))
            self.sum_shape = tuple(sum_shape)

        return (adj.sum(axes=self.sum_shape),)

def broadcast_to(a: Tensor, shape: Tuple[int]):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(
        self,
        axes: Optional[Tuple[int]] = None,
        keepdims: bool = False
    ):
        self.axes = axes
        self.keepdims = keepdims

    def compute(self, a: NDArray) -> NDArray:
        return array_api.sum(a, self.axes, keepdims=self.keepdims)

    def gradient(self, node: Tensor, adj: Tensor) -> Tensor:
        x = node.inputs[0]
        if not x.requires_grad:
            return (None,)

        if self.keepdims is True:
            return adj.broadcast_to(x.shape)

        bcast_axes = list(x.shape)
        for i in self.axes:
            bcast_axes[i] = 1

        return (adj.reshape(tuple(bcast_axes)).broadcast_to(x.shape),)

def summation(a: Tensor, axes=None) -> Tensor:
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a @ b

    def gradient(self, node: Tensor, adj: Tensor):
        lhs, rhs = node.inputs

        grad_lhs = adj @ rhs.t if lhs.requires_grad else None
        grad_rhs = lhs.t @ adj if rhs.requires_grad else None

        return grad_lhs, grad_rhs

def matmul(a: Tensor, b: Tensor) -> Tensor:
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a: NDArray) -> NDArray:
        return -1 * a

    def gradient(self, node: Tensor, adj: Tensor) -> Tensor:
        if not node.inputs[0].requires_grad:
            return (None,)

        return (-1 * adj,)

def negate(a: Tensor) -> Tensor:
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a: NDArray) -> NDArray:
        return array_api.log(a)

    def gradient(self, node: Tensor, adj: Tensor):
        x = node.inputs[0]
        if not x.requires_grad:
            return (None,)

        return (adj / x,)

def log(a: Tensor) -> Tensor:
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a: NDArray) -> NDArray:
        return array_api.exp(a)

    def gradient(self, node: Tensor, adj: Tensor) -> Tensor:
        if not node.inputs[0].requires_grad:
            return (None,)

        return (adj * node,)

def exp(a: Tensor):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a: NDArray) -> NDArray:
        return array_api.maximum(a, 0)

    def gradient(self, node: Tensor, adj: Tensor):
          x = node.inputs[0]
          if not x.requires_grad:
              return (None,)

          return adj * (x > 0)

def relu(a):
    return ReLU()(a)
