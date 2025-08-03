from __future__ import annotations
from typing import Optional, List, Tuple
from soket.ops import Op
from soket.backend.numpy import NDArray

class Node:
    """ Represents a single Node in computational graph """

    # Operator type
    op: Optional[Op]

    # Input Nodes
    inputs: List[Node]

    # Data
    cached_data: NDArray

    # Should the gradient be calculated for this node?
    requires_grad: bool

    def compute_cached_data(self) -> NDArray:
        """ Compute the cached data from the inputs.

        Returns
        -------
        output: NDArray
            Computed cached data

        """

        if self.cached_data is None:
            self.cached_data = self.op.compute(
                *[x.compute_cached_data() for x in self.inputs]
            )

        return self.cached_data

    @property
    def is_leaf(self) -> bool:
        """ Check whether this Node is leaf (input node) """
        return self.op is None

    def init(
        self,
        op: Optional[Op],
        inputs: List[Node],
        *,
        cached_data: Optional[NDArray] = None,
        requires_grad: Optional[bool] = None
    ):
        """
        Initializes and populates Node/Tensor fields.

        Parameters
        ----------
        op: Optional[Op]
            Operator type

        inputs: List[Node]
            List of inputs of atmost two Nodes

        cached_data: Optional[NDArray]
            Cached data if precomputed

        requires_grad: Optional[bool]
        """

        # If input tensors gradients are being computed, the output nodes
        # (this node) gradient should be calculated as well.
        if requires_grad is None:
            requires_grad = any(x.requires_grad for x in inputs)

        self.op = op
        self.inputs = inputs
        self.cached_data = cached_data
        self.requires_grad = requires_grad


#################################
###### GRADIENT COMPUTATION #####
#################################


def make_gradient_compatible(node: Node):
    """
    There might be some cases when operation might use broadcasting to match shapes.
    If the a node/tensor ends up getting broadcasted, there might be chances the gradient
    will not match the shape of the tensor. Use summation to make up for that.
    """

    if node.shape == node.grad.shape:
        # Nothing to do
        return

    rv_grad_axes = reversed(node.grad.shape)
    rv_node_axes = reversed(node.shape)
    grad_axes_len = len(node.grad.shape)
    node_axes_len = len(node.shape)
    diff = grad_axes_len - node_axes_len

    sum_axes = list(range(diff))
    for i, (node_axis, grad_axis) in enumerate(zip(rv_node_axes, rv_grad_axes)):
        # Assuming broadcast shape is compatible
        if node_axis != grad_axis:
            # The index is for reversed shape. Also adjust for missing
            # dimension in the node.
            sum_axes.append(diff + (node_axes_len - i - 1))

    node.grad = node.grad.sum(*sum_axes, keepdims=True)

    # Remove reduced shape if need be
    if node.grad.shape != node.shape:
        node.grad = node.grad.reshape(node.shape)


def compute_gradient(out, grad):
    """ Take gradient of the output node w.r.t. all the other node in computational graph.

    Store the computed gradient in tensor.grad field.
    """

    import gc

    # Prepare the list of topologically sorted nodes which requires gradient
    #
    sorted_list = reversed_topological_sort_and_init(out)

    # Prepare output node.
    out.grad.append(grad)

    for node in sorted_list:
        node.grad = sum_node_list(node.grad)  # This is adjoint/gradient value
        make_gradient_compatible(node)

        if not node.is_leaf:
            input_gradients = node.op.gradient(node, node.grad)

            # Accumulate partial adjoints.
            for i, g in zip(node.inputs, input_gradients):
                if i.requires_grad:
                    i.grad.append(g)

        # If the node is not being forced to store the gradient and is not a
        # leaf node, discard the gradient (and free some memory).
        if not node._force_grad and not node.is_leaf:
            del node.grad
            node.grad = None

    gc.collect()


def reversed_topological_sort_and_init(out_node: Node) -> List[Node]:
    """ A pre-order iterative DFS to find reverse topological order Node list
    of the computational graph. Also initializes tensor.grad with list() for two reasons:
     1. To accumulate partial adjoints.
     2. To mark the node as visited/processed in the topological sort.
    """

    """ The direction of the computational graph is supposed to be from input to output. The graph is being
    traversed in the reverse way using DFS, which luckily would result a topologically ordered traversal in this case. """

    s = [ (out_node, False) ]    # (node, is_processed)

    sorted_nodes: List[Node] = []
    while s:
        node, is_processed = s.pop()

        # If the node is processed.
        if is_processed:
            if not isinstance(node.grad, list):
                # Mark the node as visited
                node.grad = list()
                sorted_nodes.append(node)
        else:
            s.append((node, True))

            # Push inputs to the node
            for i in node.inputs:
                # Stage the input node if gradient needs to be computed and
                # is not visited
                if i.requires_grad and not isinstance(i.grad, list):
                    s.append((i, False))

    return reversed(sorted_nodes)


##############################
####### Helper Methods #######
##############################


def sum_node_list(node_list):
    """ Custom sum function in order to avoid create redundant nodes in Python sum implementation. """

    from operator import add
    from functools import reduce

    return reduce(add, node_list)
