from cpython.object cimport PyObject
from soket.backend cimport Device
from soket.dtype cimport DType
from soket.tensor.ops cimport *


cdef struct OpInput:
    ## Represents tensor operation inputs

    PyObject *x
    PyObject *y

# Output of backward pass
cdef struct BackwardOutput:
    PyObject *x
    PyObject *y


## TYPEDEFS ##

# Function signature type for forward computation functions
ctypedef object (*forward_fn)(
    # Target tensor object which will be built upon current operation.
    # Provided so that specific operation information can be cached which are
    # critical for backward pass.
    object target,

    # Target input x
    object x,
    int *x_shape,
    int x_nshape,

    # Target input y
    object y,
    int *y_shape,
    int y_nshape
)

ctypedef BackwardOutput (*backward_fn)(
    # Node/tensor object being differentiated w.r.t. its inputs. In other
    # words, partial adjoints are being computed for provided node inputs.
    object node,

    # Incoming gradient
    object adj,
    int *adj_shape,
    int adj_nshape,

    # Node input x
    object x,
    int *x_shape,
    int x_nshape,

    # Node input y
    object y,
    int *y_shape,
    int y_nshape
)

## TYPEDEFS END ##


cdef enum OpType:
    ## Tensor operations which builds computational graph.

    ELEMWISE_ADD = 0
    SCALAR_ADD = 1

    # Invalid operation, used to denote leaf tensor
    INVALID = -1


cdef struct Op:
    ## Represents a Tensor operation
    
    # Forward and backward function of the operation
    forward_fn fwd
    backward_fn bwd

    # Type of the operation
    OpType type


cdef class Tensor:
    ''' Represents a soket Tensor. '''

    # Object to cache the data
    cdef object _data

    # Compute device tensor belongs to
    cdef Device _device

    # Tensor datatype
    cdef DType _dtype

    # Inputs to this node/tensor in computational graph.
    cdef OpInput _inputs

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

    ## CDEF METHODS ##

    cdef void _tensor_init(
        self,
        Op op,
        OpInput inputs,
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
        requires_grad
    )

    @staticmethod
    cdef Tensor _make_from_op(
        Op op,
        OpInput inputs,
        DType dtype,
        int *shape, int nshape
    )

    ## CDEF STATIC METHODS ##

    ## CDEF OPERATOR METHODS ##

    cdef Tensor _add(self, other)

    ## CDEF OPERATOR METHODS END ##