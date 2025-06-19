from __future__ import annotations
from typing import Tuple, Union, Optional
from soket.backend.numpy import NDArray, array_api
import numpy
import soket


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


class AddScalar(TensorOp):
    def __init__(self, scalar: any):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return a + self.scalar

    def gradient(self, node: Tensor, adj: Tensor) -> Tuple[Tensor]:
        if not node.inputs[0].requires_grad:
            return (None,)
        return (adj,)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a * b

    def gradient(self, node: Tensor, adj: Tensor) -> Tuple[Tensor]:
        lhs, rhs = node.inputs
        grad_lhs = adj * rhs if lhs.requires_grad else None
        grad_rhs = adj * lhs if rhs.requires_grad else None
        return grad_lhs, grad_rhs


class MulScalar(TensorOp):
    def __init__(self, scalar: any):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return a * self.scalar

    def gradient(self, node: Tensor, adj: Tensor) -> Tuple[Tensor]:
        if not node.inputs[0].requires_grad:
            return (None,)
        return (adj * self.scalar,)


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


class EWiseDiv(TensorOp):
    """ Op to element-wise divide two nodes. """

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a / b

    def gradient(self, node: Tensor, adj: Tensor) -> Tuple[Tensor]:
        a, b = node.inputs

        grad_a = adj / a if a.requires_grad else None
        grad_b = - adj * a / (b ** 2) if b.requires_grad else None

        return grad_a, grad_b


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a) -> NDArray:
        return a / self.scalar

    def gradient(self, node: Tensor, adj: Tensor) -> Tuple[Tensor]:
        if not node.inputs[0].requires_grad:
            return (None,)
        return (adj / self.scalar,)


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

        return (adj.sum(self.sum_shape),)


class Summation(TensorOp):
    def __init__(
        self,
        axes: Optional[Tuple[int]] = None,
        keepdims: Optional[bool] = False
    ):
        self.axes = axes
        self.keepdims = keepdims

    def compute(self, a: NDArray) -> NDArray:
        return a.sum(self.axes, keepdims=self.keepdims)

    def gradient(self, node: Tensor, adj: Tensor) -> Tensor:
        x = node.inputs[0]
        if not x.requires_grad:
            return (None,)

        if self.keepdims is True:
            return adj.broadcast_to(x.shape)

        bcast_axes = list(x.shape)
        if self.axes is not None:
            for i in self.axes:
                bcast_axes[i] = 1
        else:
            bcast_axes = [1] * len(x.shape)

        return (adj.reshape(tuple(bcast_axes)).broadcast_to(x.shape),)


class MatMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a @ b

    def gradient(self, node: Tensor, adj: Tensor):
        lhs, rhs = node.inputs

        grad_lhs = adj @ rhs.t if lhs.requires_grad else None
        grad_rhs = lhs.t @ adj if rhs.requires_grad else None

        return grad_lhs, grad_rhs


class Negate(TensorOp):
    def compute(self, a: NDArray) -> NDArray:
        return -a

    def gradient(self, node: Tensor, adj: Tensor) -> Tensor:
        if not node.inputs[0].requires_grad:
            return (None,)

        return (-adj,)


class Log(TensorOp):
    def compute(self, a: NDArray) -> NDArray:
        return array_api.log(a)

    def gradient(self, node: Tensor, adj: Tensor):
        x = node.inputs[0]
        if not x.requires_grad:
            return (None,)

        return (adj / x,)


class Exp(TensorOp):
    def compute(self, a: NDArray) -> NDArray:
        return array_api.exp(a)

    def gradient(self, node: Tensor, adj: Tensor) -> Tensor:
        if not node.inputs[0].requires_grad:
            return (None,)

        return (adj * node,)


class ReLU(TensorOp):
    def compute(self, a: NDArray) -> NDArray:
        return array_api.maximum(a, 0)

    def gradient(self, node: Tensor, adj: Tensor):
          x = node.inputs[0]
          if not x.requires_grad:
              return (None,)

          return (adj * (x > 0),)


class LogSumExp(Summation):
    def __init__(
        self,
        axes: Optional[Tuple[int]] = None,
        keepdims: Optional[bool] = False
    ):
        super().__init__(axes, keepdims)

    def compute(self, z: NDArray) -> NDArray:
        max = z.max(axis=self.axes, keepdims=True)
        res = array_api.log(array_api.exp(z - max).
            sum(axis=self.axes, keepdims=True)) + max

        return res.reshape(tuple([x for x in res.shape if x != 1]))    \
            if not self.keepdims else res
    
    def gradient(self, node: Tensor, adj: Tensor) -> Tensor:
        z = node.inputs[0]
        max = soket.Tensor.make_const(
            data=z.compute_cached_data().max(axis=self.axes, keepdims=True),
            requires_grad=True
        )
        exp_z = soket.exp(z - max)
        sum_exp_z = exp_z.sum(*self.axes, keepdims=True)
        return (super().gradient(node, adj)[0] * exp_z / sum_exp_z,)


class SoftmaxCrossEntropy(LogSumExp):
    def __init__(
        self,
        axes: Optional[Tuple[int]],
        y: Tensor,
        num_classes: int,
        reduction: str = 'mean'
    ):
        super().__init__(axes)

        # Save one hot version of the ground truth
        self.one_hot = soket.one_hot(num_classes, y)
        self.reduction = reduction

    def compute(self, Z: NDArray) -> NDArray:
        """ Z: The logits of dimension (B, O), where B is the batch size
        and O is the input size.

        y: The label array (1d tensor) of size (B,) only.
        """
    
        batch_xentropy = super().compute(Z) - (Z * self.one_hot.compute_cached_data()) \
            .sum(self.axes)
        
        if self.reduction == 'mean':
            return batch_xentropy.mean()
        elif self.reduction == 'sum':
            return batch_xentropy.sum()
        
        return batch_xentropy
    
    def gradient(self, node: Tensor, adj: Tensor):
        """
            Note: Gradient for 'none' reduction type will result same gradient
            as 'sum' reduction. This is because loss functions should always do
            some sort of reduction to a scalar value.
        """

        z = node.inputs[0]
        exp_z = soket.exp(z)
        sum_exp_z = exp_z.sum(self.axes, keepdims=True)

        batch_grad = (exp_z / sum_exp_z) - self.one_hot

        if self.reduction == 'mean':
            batch_size = self.one_hot.shape[-2]
            return (batch_grad / batch_size,)
        elif self.reduction == 'none':
            new_shape = list(adj.shape) + [1]
            adj = adj.reshape(tuple(new_shape))
        
        return (adj * batch_grad,)
    

class Select(TensorOp):
    def __init__(self, idx: int | slice | Tuple[int | slice]):
        self.idx = idx
    
    def compute(self, a: NDArray):
        return a.__getitem__(self.idx) if array_api is numpy else a.get(self.idx)
    
    def gradient(self, node: Tensor, adj: Tensor) -> Tensor:
        a = node.inputs[0]
        grad = soket.zeros_like(a)
        grad.__setitem__(self.idx, adj)

        return (grad,)