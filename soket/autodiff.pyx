from cpython.ref cimport Py_XDECREF
from cpython.object cimport PyObject, Py_TYPE, PyTypeObject
from cpython.list cimport (PyList_New, PyList_Append, PyList_GET_ITEM,
    PyList_GET_SIZE)
from cpython.tuple cimport (PyTuple_New, PyTuple_SET_ITEM)
from soket.tensor cimport Tensor, OpType, OpInput, BackwardOutput
from soket.tensor.ops.intern cimport _sum, _reshape
from libc.stdlib cimport malloc, realloc, free
from libc.string cimport memcmp, memcpy


## DEFINES ##

DEF _TOPOLOGICAL_SORT_STACK_SIZE = 1024

## DEFINES END ##


## HELPER FUNCTIONS ##

cdef inline int _PUSH((PyObject *, bint) *stack, int top, (PyObject *, bint) value):
    ''' Topological sort helper function to push value on to the stack. '''

    if top + 1 >= _TOPOLOGICAL_SORT_STACK_SIZE:
        raise RuntimeError('Topological sort stack overflow!')
    stack[top] = value

    return top + 1

cdef inline Tensor _sum_nodes(list nodes):
    ''' Sums up the partial adjoints to create adjoint. '''

    cdef Py_ssize_t size = PyList_GET_SIZE(nodes)
    if size == 0:  # Unlikely
        raise RuntimeError('Input node list is empty! Check for AD issues.')

    cdef Tensor res = <Tensor> PyList_GET_ITEM(nodes, 0)
    for i in range(1, size):
        res = res._add(<Tensor> PyList_GET_ITEM(nodes, i))

    return res

cdef inline void _make_gradient_compatible(Tensor node):
    '''
    There might be cases where input shapes might get broadcasted. Consider
    this scenario:

      z = x * y, where x_shape = (1, 5, 3), y_shape = (5, 5, 1)

    Here, both of the shapes are getting broadcasted. The output shape is
    (5, 5, 3). Now, when adjoint/gradient is calculated for both x and y,
    the shape of the gradient is (5, 5, 3) in both cases, which is
    incompatible. To make up for that, summation reduction is performed (
    alongside with reshaping). For this example, if summation is performed
    on the gradient of x in first dimension (with keepdims) and in last
    for gradient y (with keepdims), the gradients shape would be compatible.
    '''

    cdef Tensor grad = <Tensor> node._grad

    # If gradient shape is compatible, then nothing to do.
    if (node._nshape == grad._nshape and
    not memcmp(node._shape, grad._shape, node._nshape * sizeof(int))):
        return

    # grad._nshape > node._nshape always
    cdef int diff = grad._nshape - node._nshape
    cdef int sum_axes_size = diff

    cdef Py_ssize_t i
    for i in range(node._nshape):
        if grad._shape[i + diff] != node._shape[i]:
            sum_axes_size += 1

    # Prepare summation axes
    cdef tuple sum_axes = PyTuple_New(sum_axes_size)
    cdef int sum_axes_idx = 0
    for i in range(grad._nshape):
        if i < diff or grad._shape[i] != node._shape[i - diff]:
            PyTuple_SET_ITEM(sum_axes, sum_axes_idx, i)
            sum_axes_idx += 1

    cdef object data = grad._compute_data()
    data = _sum(grad._device._dev_idx)(data, sum_axes, None, None, True)

    if node._nshape != grad._nshape:
        data = _reshape(grad._device._dev_idx)(
            data,
            node._compute_data().shape
        )

    ## Reconstruct gradient tensor

    # New shape
    free(grad._shape)
    grad._shape = <int *> malloc(node._nshape * sizeof(int))
    memcpy(grad._shape, node._shape, node._nshape * sizeof(int))
    grad._nshape = node._nshape

    # Finally, update the data
    grad._data = data

## HELPER FUNCTIONS END ##


cdef void _compute_gradient(Tensor node, Tensor adj):
    '''
    Take gradient of `node` w.r.t. all the other nodes in the computational
    graph and store it in Tensor._grad field.
    '''

    # Python scope forward declaration
    cdef Tensor n, in_x, in_y
    cdef OpInput inputs
    cdef BackwardOutput bw_output

    # Perform topological sort.
    cdef PyObject **sorted_nodes
    cdef int sorted_nodes_len
    sorted_nodes, sorted_nodes_len = _topological_sort_and_init(node)

    # Prepare given node
    PyList_Append(node._grad, adj)

    for i in range(sorted_nodes_len - 1, -1, -1):
        n = <Tensor> sorted_nodes[i]

        # Sum the partial gradients
        n._grad = _sum_nodes(<list> n._grad)
        _make_gradient_compatible(n)

        # If not a leaf tensor
        if n._op.type != OpType.INVALID:
            inputs = n._inputs
            in_x = <Tensor> inputs.x if inputs.x != NULL else <Tensor> None
            in_y = <Tensor> inputs.y if inputs.y != NULL else <Tensor> None

            bw_output = n._op.bwd(
                n, <Tensor> n._grad,
                in_x, in_y
            )

            # Accumulate partial adjoints
            if bw_output.x != NULL and in_x._requires_grad:
                PyList_Append(in_x._grad, <object> bw_output.x)

            if bw_output.y != NULL and in_y._requires_grad:
                PyList_Append(in_y._grad, <object> bw_output.y)

            # If the node is not forced to store the gradient and is not a
            # leaf node/tensor, discard the gradient (and hopefully free some
            # memory).
            if not n._retain_grad:
                n._grad = None

            # Objects in BackwardOutput structure has manually increased
            # reference count. Decrease to balance.
            Py_XDECREF(bw_output.x)
            Py_XDECREF(bw_output.y)

    # Allocated in `_topological_sort_and_init`
    free(sorted_nodes)


cdef (PyObject **, int) _topological_sort_and_init(Tensor node):
    '''
    Returns a Tensor/Node sequence denoting a topological sort of the
    computational graph, performing pre-order iterative DFS.

    This function also initialized Tensor._grad field with list() for three
    reasons:
     1. To denote the node has been processed/visited.
     2. To accumulate partial adjoints.
     3. Manual memory management would be tedious as we don't know the exact
        amount of partial adjoints being worked on for the particular node.

    But it is ensured that after the gradient computation is done, the
    Tensor._grad field will hold a Tensor denoting a node's gradient.
    '''

    # Python scope forward declaration
    cdef PyObject *n
    cdef bint is_processed
    cdef OpInput inputs
    cdef Tensor ten

    # Sorted nodes list.
    cdef int sorted_nodes_size = 64
    cdef int sorted_nodes_idx = 0
    cdef PyObject **sorted_nodes = <PyObject **> malloc(
        sorted_nodes_size * sizeof(PyObject *)
    )
    if sorted_nodes == NULL:
        raise MemoryError(
            'Failed to allocate memory for topological sort buffer!'
        )

    # (node, is_processed)
    cdef (PyObject *, bint)[_TOPOLOGICAL_SORT_STACK_SIZE] stack
    cdef int top = 0

    # Start with the given node
    stack[top] = (<PyObject *> node, False)
    top += 1

    while top > 0:
        # pop
        top -= 1
        n, is_processed = stack[top]

        # If the node is processed
        if is_processed:
            ten = <Tensor> n

            # If the node is not initialized/visited
            if Py_TYPE(ten._grad) is not <PyTypeObject *> list:
                ten._grad = PyList_New(0)

                # If sorted node list overflows.
                if sorted_nodes_idx + 1 >= sorted_nodes_size:
                    sorted_nodes_size *= 2
                    sorted_nodes = <PyObject **> realloc(
                        sorted_nodes,
                        sorted_nodes_size
                    )
                    if sorted_nodes == NULL:
                        raise RuntimeError(
                            'Could not reallocate sorted list memory!'
                        )

                sorted_nodes[sorted_nodes_idx] = n
                sorted_nodes_idx += 1
        else:
            top = _PUSH(stack, top, (n, True))

            # Push node inputs
            inputs = (<Tensor> n)._inputs
            if inputs.x != NULL:
                ten = <Tensor> inputs.x
                if (ten._requires_grad and Py_TYPE(ten._grad) is
                not <PyTypeObject *> list):
                    top = _PUSH(stack, top, (inputs.x, False))

            if inputs.y != NULL:
                ten = <Tensor> inputs.y
                if (ten._requires_grad and Py_TYPE(ten._grad) is
                not <PyTypeObject *> list):
                    top = _PUSH(stack, top, (inputs.y, False))

    return (sorted_nodes, sorted_nodes_idx)
