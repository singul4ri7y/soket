from cpython.object cimport PyObject, Py_TYPE, PyTypeObject
from cpython.tuple cimport PyTuple_GET_SIZE, PyTuple_GET_ITEM
from soket.backend cimport Device
from soket.dtype cimport DType
from libc.stdlib cimport malloc, free, qsort
cimport numpy as np


cdef struct TensorTriad:
    # Generally, main focus should be these two tensors:
    PyObject *x
    PyObject *y

    # Extra third tensor
    PyObject *Z

cdef struct _ShapeInfo:
    # Contains shape information.
    # The `shape` pointer is heap allocated, use with caution.

    int *shape
    int nshape


## TYPEDEFS ##

# Function signature type for forward computation functions
ctypedef object (*forward_fn)(
    # Target tensor object which will be built upon current operation.
    # Some operation requires access to some cached value (e.g. a scalar for
    # scalar addition).
    Tensor target,

    # Target input x
    object x,

    # Target input y
    object y,

    # Target (extra) input Z
    object Z
)

ctypedef TensorTriad (*backward_fn)(
    # Node/tensor object being differentiated w.r.t. its inputs. In other
    # words, partial adjoints are being computed for provided node inputs.
    Tensor node,

    # Incoming gradient
    Tensor adj,

    # Node input x
    Tensor x,

    # Node input y
    Tensor y,

    # Node (extra) input Z
    Tensor Z
)

## TYPEDEFS END ##


cdef enum OpType:
    ## Tensor operations which builds computational graph.

    ELEMWISE_ADD = 0,
    SCALAR_ADD,
    NEGATE,
    ELEMWISE_SUB,
    SCALAR_SUB,
    ELEMWISE_MUL,
    SCALAR_MUL,
    ELEMWISE_DIV,
    SCALAR_DIV,
    ELEMWISE_POW,
    SCALAR_POW,
    BROADCAST_TO,
    SUMMATION,
    ABS,
    MEAN,
    STD,
    VAR,
    MAX,
    MIN,
    MATMUL,
    RESHAPE,
    PERMUTE,
    TRANSPOSE,
    SELECT,
    LOG,
    EXP,
    LOGSUMEXP,

    # Loss operations
    SXENTROPYLOSS,

    # Normalization operations
    BATCHNORM,

    # Acvitation operations
    RELU,
    TANH,

    # Invalid operation, used to denote leaf tensor
    INVALID = -1


cdef struct Op:
    ## Represents a Tensor operation

    # Forward and backward function of the operation
    forward_fn fwd
    backward_fn bwd

    # Type of the operation
    OpType type


## HELPER FUNCTIONS ##

cdef inline _ShapeInfo _get_shape_info_from_tuple(shape):
    '''
    Extracts shape in a contiguous integer array from tuple.

    The tuple is expected to contain only integers.
    '''

    cdef _ShapeInfo info
    info.nshape = PyTuple_GET_SIZE(shape)

    # Allocate shape
    info.shape = <int *> malloc(info.nshape * sizeof(int))
    if info.shape is NULL:
        raise MemoryError(f'Could not allocate memory to store shape!')

    for i in range(info.nshape):
        info.shape[i] = <int> <object> PyTuple_GET_ITEM(shape, i)

    return info


cdef inline tuple _get_proper_shape(tuple shape):
    ''' Get the preferred shape. '''

    cdef int len = PyTuple_GET_SIZE(shape)
    cdef object first = None
    if len > 0:
        first = <object> PyTuple_GET_ITEM(shape, 0)

    if (Py_TYPE(first) is <PyTypeObject *> tuple or
    Py_TYPE(first) is <PyTypeObject *> list):
        shape = tuple(first)
        len = PyTuple_GET_SIZE(shape)

    for i in range(len):
        if (Py_TYPE(<object> PyTuple_GET_ITEM(shape, i)) is not
        <PyTypeObject *> int):
            raise RuntimeError('Expected the shape to contain only integers!')

    return shape

cdef inline tuple _get_proper_reduction_axes(tuple axes):
    ''' Get dominating reduction axes. '''

    cdef int len = PyTuple_GET_SIZE(axes)
    cdef object first = None
    if len > 0:
        first = <object> PyTuple_GET_ITEM(axes, 0)

    if (Py_TYPE(first) is <PyTypeObject *> tuple or
    Py_TYPE(first) is <PyTypeObject *> list):
        axes = tuple(first)
        len = PyTuple_GET_SIZE(axes)

    for i in range(len):
        if (Py_TYPE(<object> PyTuple_GET_ITEM(axes, i)) is not
        <PyTypeObject *> int):
            raise RuntimeError('Expected the axes to contain only integers!')

    return axes


cdef inline int _ascending_sort(const void *a, const void *b) noexcept nogil:
    return (<int *> a)[0] - (<int *> b)[0]

cdef inline (int *, int) _validate_and_create_reduction_axes(
    tuple axes,
    int *shape,
    int nshape,
    bint keepdims
):
    ''' Validates a list of reduction axes. '''

    cdef int axes_len = <int> PyTuple_GET_SIZE(axes)

    # Reduce over all axes
    if axes_len == 0:
        return <int *> malloc(0), 0

    cdef int *_axes = <int *> malloc(axes_len * sizeof(int))
    cdef int new_nshape = nshape if keepdims else nshape - axes_len
    cdef int *new_shape = <int *> malloc(new_nshape * sizeof(int))
    if new_shape == NULL:
        free(_axes)
        raise MemoryError('Failed to allocate memory to store axes!')

    cdef object elem
    cdef int axis

    for i in range(axes_len):
        elem = <object> PyTuple_GET_ITEM(axes, i)
        axis = <int> elem

        if axis < -nshape or axis >= nshape:
            free(_axes)
            free(new_shape)
            raise ValueError(f'Reduction axis out of bounds: {elem}')

        _axes[i] = axis + nshape if axis < 0 else axis

    # Sort axes
    qsort(_axes, axes_len, sizeof(int), _ascending_sort)

    cdef int axes_idx = 0
    cdef int new_idx = 0
    for i in range(nshape):
        if axes_idx < axes_len and i == _axes[axes_idx]:
            axes_idx += 1

            if keepdims:
                new_shape[new_idx] = 1
                new_idx += 1
            else: continue
        else:
            new_shape[new_idx] = shape[i]
            new_idx += 1

    free(_axes)
    return new_shape, new_nshape

## HELPER FUNCTIONS END ##


cdef class Tensor:
    ''' Represents a soket Tensor. '''

    # Object to cache the data
    cdef object _data_

    # Compute device tensor belongs to
    cdef Device _device

    # Tensor datatype
    cdef DType _dtype

    # Inputs to this node/tensor in computational graph.
    cdef TensorTriad _inputs

    # Ensured to store the gradient/adjoint. May used to hold list of partial
    # adjoints during gradient computation.
    cdef object _grad

    # Tensor shape
    cdef int *_shape

    # Operation performed to create this tensor
    cdef Op _op

    # Tensor number of dimensions
    cdef int _nshape

    # Should gradient be computed for this node?
    cdef bint _requires_grad

    # Should computed gradient be stored even if the Node/Tensor is not leaf?
    cdef bint _retain_grad

    # For caching values, useful for both forward and backward pass
    # TODO: ADD NULL AT THE END TO TERMINATE VALUE_CACHE WHEN BUILDING FROM
    #       TENSOR.
    cdef int _n_value_cache
    cdef PyObject **_value_cache

    ## METHODS ##

    cpdef Tensor to(self, Device device)
    cpdef Tensor copy(self, requires_grad=?)
    cpdef void backward(self, Tensor adj=?)
    cpdef Tensor abs(self)
    cpdef np.ndarray numpy(self)

    ## METHODS END ##

    ## CDEF METHODS ##

    cdef void _tensor_init(
        self,
        Op op,
        TensorTriad inputs,
        object data,
        Device device,
        DType dtype,
        int *shape, int nshape,
        bint copy_shape,
        requires_grad
    )

    cdef object _compute_data(self)

    ## CDEF METHODS ##

    ## CDEF STATIC METHODS ##

    @staticmethod
    cdef Tensor _make_const(
        data,
        Device device,
        DType dtype,
        int *shape, int nshape,
        bint copy_shape,
        requires_grad
    )

    @staticmethod
    cdef Tensor _make_from_op(
        Op op,
        TensorTriad inputs,
        Device device,
        DType dtype,
        int *shape, int nshape,
        bint copy_shape,
        PyObject **cache, int ncache
    )

    @staticmethod
    cdef Tensor _from_numpy(object array)

    ## CDEF STATIC METHODS ##

    ## CDEF OPERATOR METHODS ##

    cdef Tensor _add(self, other: Tensor | any)
    cdef Tensor _neg(self)
    cdef Tensor _sub(self, other: Tensor | any)
    cdef Tensor _rsub(self, other: Tensor | any)
    cdef Tensor _mul(self, other: Tensor | any)
    cdef Tensor _div(self, other: Tensor | any)
    cdef Tensor _rdiv(self, other: Tensor | any)
    cdef Tensor _pow(self, other: Tensor | any)
    cdef Tensor _rpow(self, other: Tensor | any)
    cdef Tensor _broadcast_to(self, tuple other)
    cdef Tensor _sum(self, tuple axes, DType dtype, keepdims)
    cdef Tensor _mean(self, tuple axes, DType dtype, keepdims)
    cdef Tensor _std(self, tuple axes, DType dtype, keepdims, correction)
    cdef Tensor _var(self, tuple axes, DType dtype, keepdims, correction)
    cdef Tensor _max(self, tuple axes, DType dtype, keepdims)
    cdef Tensor _min(self, tuple axes, DType dtype, keepdims)
    cdef Tensor _matmul(self, Tensor other)
    cdef Tensor _reshape(self, tuple shape)
    cdef Tensor _permute(self, tuple axes)
    cdef Tensor _transpose(self)
    cdef Tensor _getitem(self, idx: int | slice | tuple[int | slice])
    cdef Tensor _argmax(self, axis, keepdims)
    cdef Tensor _argmin(self, axis, keepdims)
    cdef void _setitem(
        self,
        idx: int | slice | tuple[int | slice],
        value: Tensor | any
    )
    cdef Tensor _detach(self)
    cdef Tensor _data(self)
    cdef void _set_data(self, Tensor data)

    ## CDEF OPERATOR METHODS END ##

    ## CDEF COMPARISON OPERATORS ##

    cdef Tensor _eq(self, other: Tensor | any)
    cdef Tensor _ne(self, other: Tensor | any)
    cdef Tensor _gt(self, other: Tensor | any)
    cdef Tensor _rgt(self, other: Tensor | any)
    cdef Tensor _ge(self, other: Tensor | any)
    cdef Tensor _rge(self, other: Tensor | any)
    cdef Tensor _lt(self, other: Tensor | any)
    cdef Tensor _rlt(self, other: Tensor | any)
    cdef Tensor _le(self, other: Tensor | any)
    cdef Tensor _rle(self, other: Tensor | any)

    ## CDEF COMPARISON OPERATORS END ##
