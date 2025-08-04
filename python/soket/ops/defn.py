from __future__ import annotations
from typing import Tuple, Union, Optional
from abc import ABC, abstractmethod
from soket.backend import NDArray
import soket
import math


class Op(ABC):
    """ Operator class, works as a interface class to tensor specific
    operator classes

    """

    @abstractmethod
    def __call__(self, *args):
        raise NotImplementedError()

    @abstractmethod
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

    @abstractmethod
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


class TensorOp(Op):
    """ Oeperator class geared towards Tensors. """

    def __call__(self, *args):
        return soket.Tensor._make_from_op(self, args)


## TENSOR OPERATIONS ##

class EWiseAdd(TensorOp):
    """ Element-wise add operation w/o scalar. """

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a + b

    def gradient(self, node: Tensor, adj: Tensor) -> Tuple[Tensor]:
        a, b = node.inputs
    
        grad_a = adj if a.requires_grad else None
        grad_b = adj if b.requires_grad else None

        return grad_a, grad_b


class AddScalar(TensorOp):
    """ Tensor element-wise addition with a scalar. """

    def __init__(self, scalar: any):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return a + self.scalar

    def gradient(self, node: Tensor, adj: Tensor) -> Tuple[Tensor]:
        if not node.inputs[0].requires_grad:
            return (None,)
        
        return (adj,)


class EWiseSub(TensorOp):
    """ Element-wise subtraction operation w/o scalar. """

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a - b

    def gradient(self, node: Tensor, adj: Tensor) -> Tuple[Tensor]:
        a, b = node.inputs
    
        grad_a = adj if a.requires_grad else None
        grad_b = -adj if b.requires_grad else None

        return grad_a, grad_b


class SubScalar(TensorOp):
    """ Tensor element-wise subtraction with a scalar. """

    def __init__(self, scalar: any, commute: bool = False):
        self.scalar = scalar
        self.commute = commute

    def compute(self, a: NDArray) -> NDArray:
        return a - self.scalar if not self.commute  \
            else self.scalar - a

    def gradient(self, node: Tensor, adj: Tensor) -> Tuple[Tensor]:
        if not node.inputs[0].requires_grad:
            return (None,)
        
        return (adj if not self.commute else -adj,)


class EWiseMul(TensorOp):
    """ Tensor element-wise multiplication with a scalar. """

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a * b

    def gradient(self, node: Tensor, adj: Tensor) -> Tuple[Tensor]:
        lhs, rhs = node.inputs

        grad_lhs = adj * rhs if lhs.requires_grad else None
        grad_rhs = adj * lhs if rhs.requires_grad else None
    
        return grad_lhs, grad_rhs


class MulScalar(TensorOp):
    """ Tensor element-wise multiplication with a scalar. """

    def __init__(self, scalar: any):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return a * self.scalar

    def gradient(self, node: Tensor, adj: Tensor) -> Tuple[Tensor]:
        if not node.inputs[0].requires_grad:
            return (None,)
    
        return (adj * self.scalar,)


class EWiseDiv(TensorOp):
    """ Tensor element-wise division by another tensor. """

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a / b

    def gradient(self, node: Tensor, adj: Tensor) -> Tuple[Tensor]:
        a, b = node.inputs

        grad_a = adj / a if a.requires_grad else None
        grad_b = -adj * a * (b ** -2) if b.requires_grad else None

        return grad_a, grad_b


class DivScalar(TensorOp):
    """ Tensor element-wise division with a scalar. """

    def __init__(self, scalar: any, commute: bool = False):
        self.scalar = scalar
        self.commute = commute

    def compute(self, a) -> NDArray:
        one_over_scalar = 1 / self.scalar

        return (a * one_over_scalar) if not self.commute  \
            else (self.scalar / a)

    def gradient(self, node: Tensor, adj: Tensor) -> Tuple[Tensor]:
        a = node.inputs[0]

        if not a.requires_grad:
            return (None,)
    
        one_over_scalar = 1 / self.scalar
        grad = one_over_scalar if not self.commute  \
            else -self.scalar * (a ** -2)
    
        return (grad * adj,)


class EWisePow(TensorOp):
    """ Tensor element-wise raise to a power operation. """

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a ** b

    def gradient(self, node: Tensor, adj: Tensor) -> Tuple[Tensor]:
        a, b = node.inputs

        grad_a = b * (a ** (b - 1)) if a.requires_grad else None
        grad_b = node * soket.log(a)  \
            if b.requires_grad else None

        return grad_a * adj, grad_b * adj


class PowerScalar(TensorOp):
    """ Tensor element-wise raise to a scalar power operation. """

    def __init__(self, scalar: any, commute: bool=False):
        self.scalar = scalar
        self.commute = commute

    def compute(self, a: NDArray) -> NDArray:
        return a ** self.scalar if not self.commute  \
            else self.scalar ** a

    def gradient(self, node: Tensor, adj: Tensor) -> Tuple[Tensor]:
        a = node.inputs[0]
        if not a.requires_grad:
            return (None,)
        
        grad = self.scalar * (a ** (self.scalar - 1)) if not self.commute  \
            else node * math.log(self.scalar)

        return (grad * adj,)


class Permute(TensorOp):
    """ Re-arrange dimensions of a tensor """

    def __init__(self, axes: Tuple[int]):
        self.axes = axes

    def compute(self, a: NDArray) -> NDArray:
        return NDArray.permute(a, self.axes)

    def gradient(self, node: Tensor, adj: Tensor) -> Tuple[Tensor]:
        if not node.inputs[0].requires_grad:
            return (None,)

        # Compute the inverse axes if not set
        inv_axes = [0] * len(self.axes)
        for i, x in self.axes:
            self.inv_axes[x] = i

        return (adj.permute(inv_axes),)


class Transpose(TensorOp):
    """ Only re-arrange last two dimensions (matrix transpose). """

    def compute(self, a: NDArray) -> NDArray:
        return NDArray.swapaxes(a, -1, -2)

    def gradient(self, node: Tensor, adj: Tensor) -> Tensor:
        return adj.T


class Reshape(TensorOp):
    """ Reshape a tensor. """

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
    """ Broadcast a tensor to given shape. """

    def __init__(self, shape: Tuple[int]):
        self.shape = shape

    def compute(self, a: NDArray) -> NDArray:
        return NDArray.broadcast_to(a, self.shape)

    def gradient(self, node: Tensor, adj: Tensor) -> Tensor:
        x = node.inputs[0]
        if not x.requires_grad:
            return (None,)

        # Compute the summation shape
        rv_bcast_axes = reversed(self.shape)
        rv_input_axes = reversed(x.shape)
        bcast_axes_len = len(self.shape)
        input_axes_len = len(x.shape)
        diff = bcast_axes_len - input_axes_len

        sum_shape = list(range(diff))
        for i, (input_shape, bcast_shape) in  \
        enumerate(zip(rv_bcast_axes, rv_input_axes)):
            # Assuming broadcast shape is compatible
            if input_shape != bcast_shape:
                # The index is for reversed shape. Also adjust for missing
                # dimension in the input tensor.
                sum_shape.append(diff + (input_axes_len - i - 1))

        grad = adj.sum(sum_shape, keepdims=True)
        if grad.shape != x.shape:
            grad = grad.reshape(x.shape)

        return (grad,)


class Summation(TensorOp):
    """ Reduce a tensor by performing summation. """

    def __init__(
        self,
        axes: Optional[Tuple[int]] = None,
        dtype: str = None,
        keepdims: Optional[bool] = False
    ):
        self.axes = axes
        self.dtype = dtype
        self.keepdims = keepdims

    def compute(self, a: NDArray) -> NDArray:
        return a.sum(self.axes, dtype=self.dtype, keepdims=self.keepdims)

    def gradient(self, node: Tensor, adj: Tensor) -> Tensor:
        x = node.inputs[0]
        if not x.requires_grad:
            return (None,)

        if self.keepdims is True:
            return (adj.broadcast_to(x.shape),)

        bcast_axes = list(x.shape)
        if self.axes is not None:
            for i in self.axes:
                bcast_axes[i] = 1
        else:
            bcast_axes = [1] * len(x.shape)

        return (adj.reshape(tuple(bcast_axes)).broadcast_to(x.shape),)


class Mean(Summation):
    """ Reduce a tensor by performing mean. """

    def __init__(
        self,
        observations: int,
        axes: Optional[Tuple[int]] = None,
        dtype: str = None,
        keepdims: Optional[bool] = False
    ):
        super().__init__(axes, dtype, keepdims)
        self.observations = observations

    def compute(self, a: NDArray) -> NDArray:
        return a.mean(self.axes, dtype=self.dtype, keepdims=self.keepdims)

    def gradient(self, node: Tensor, adj: Tensor) -> Tensor:
        one_over_obs = 1 / self.observations
    
        return (super().gradient(node, adj)[0] * one_over_obs,)


class MatMul(TensorOp):
    """ Matrix-matrix multiplication operation of tensors. """

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a @ b

    def gradient(self, node: Tensor, adj: Tensor):
        lhs, rhs = node.inputs

        grad_lhs = adj @ rhs.T if lhs.requires_grad else None
        grad_rhs = lhs.T @ adj if rhs.requires_grad else None

        return grad_lhs, grad_rhs


class Negate(TensorOp):
    """ Negation operation on a tensor. """

    def compute(self, a: NDArray) -> NDArray:
        return -a

    def gradient(self, node: Tensor, adj: Tensor) -> Tensor:
        if not node.inputs[0].requires_grad:
            return (None,)

        return (-adj,)


class Log(TensorOp):
    """ Perform element-wise natural log. """

    def compute(self, a: NDArray) -> NDArray:
        return NDArray.log(a)

    def gradient(self, node: Tensor, adj: Tensor):
        x = node.inputs[0]
        if not x.requires_grad:
            return (None,)

        return (adj / x,)


class Exp(TensorOp):
    """ Perform Euler's exponential operation. """

    def compute(self, a: NDArray) -> NDArray:
        return NDArray.exp(a)

    def gradient(self, node: Tensor, adj: Tensor) -> Tensor:
        if not node.inputs[0].requires_grad:
            return (None,)

        return (adj * node,)


class ReLU(TensorOp):
    """ Perform ReLU activation. """

    def compute(self, a: NDArray) -> NDArray:
        return NDArray.maximum(a, 0)

    def gradient(self, node: Tensor, adj: Tensor):
          x = node.inputs[0]
          if not x.requires_grad:
              return (None,)

          return (adj * (x > 0),)
    

class Max(TensorOp):
    """ Find maximum values over given dimensions. """

    def __init__(
        self,
        axes: Optional[Tuple[int]],
        keepdims: Optional[bool] = False
    ):
        self.axes = axes
        self.keepdims = keepdims

        # Will be used if keepdims is False.
        self._retained_dim = None
    
    def compute(self, x: NDArray) -> NDArray:
        if self.keepdims is False:
            # Calculate reduced dimensions with same shape
            self._retained_dim = list(x.shape)
            
            for axis in self.axes:
                self._retained_dim[axis] = 1

        return x.max(axes=self.axes, keepdims=self.keepdims)
    
    def gradient(self, node: Tensor, adj: Tensor) -> Tensor:
        x = node.inputs[0]

        # Calculate mask, where maximum values are found.
        if self.keepdims is False:
            node = node.reshape(self._retained_dim)
            adj = adj.reshape(self._retained_dim)
        
        mask = (x == node)
        return (adj * mask,)
    
class Min(TensorOp):
    """ Find minimum values over given dimensions. """

    def __init__(
        self,
        axes: Optional[Tuple[int]],
        keepdims: Optional[bool] = False
    ):
        self.axes = axes
        self.keepdims = keepdims

        # Will be used if keepdims is False.
        self._retained_dim = None
    
    def compute(self, x: NDArray) -> NDArray:
        if self.keepdims is False:
            # Calculate reduced dimensions with same shape
            self._retained_dim = list(x.shape)
            
            for axis in self.axes:
                self._retained_dim[axis] = 1

        return x.min(axes=self.axes, keepdims=self.keepdims)
    
    def gradient(self, node: Tensor, adj: Tensor) -> Tensor:
        x = node.inputs[0]

        # Calculate mask, where maximum values are found.
        if self.keepdims is False:
            node = node.reshape(self._retained_dim)
            adj = adj.reshape(self._retained_dim)
        
        mask = (x == node)
        return (adj * mask,)


class LogSumExp(Summation):
    def __init__(
        self,
        axes: Optional[Tuple[int]] = None,
        keepdims: Optional[bool] = False
    ):
        super().__init__(axes, keepdims)

    def compute(self, z: NDArray) -> NDArray:
        max = z.max(axes=self.axes, keepdims=True)

        return NDArray.log(NDArray.exp(z - max).
            sum(axes=self.axes, keepdims=self.keepdims)) + max

    def gradient(self, node: Tensor, adj: Tensor) -> Tensor:
        z = node.inputs[0]

        max = z.max(*self.axes, keepdims=True)
        exp_z = soket.exp(z - max)
        sum_exp_z = exp_z.sum(() if self.axes is None else self.axes, keepdims=True)

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
        self.one_hot = soket.one_hot(y, num_classes)
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
            # TODO: Make the solution more flexlible
            new_shape = list(adj.shape) + [1]
            adj = adj.reshape(tuple(new_shape))

        return (adj * batch_grad,)


class Select(TensorOp):
    def __init__(self, idx: int | slice | Tuple[int | slice]):
        self.idx = idx

    def compute(self, a: NDArray):
        return a.__getitem__(self.idx)

    def gradient(self, node: Tensor, adj: Tensor) -> Tensor:
        a = node.inputs[0]

        grad = soket.zeros_like(a)
        grad.__setitem__(self.idx, adj)

        return (grad,)
