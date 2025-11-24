from cpython.ref cimport Py_XINCREF, Py_XDECREF
from cpython.object cimport Py_TYPE, PyTypeObject
from cpython.tuple cimport PyTuple_GET_SIZE, PyTuple_GET_ITEM, PyTuple_SET_ITEM
from cpython.long cimport PyLong_FromLong, PyLong_AsLong
from cpython.float cimport PyFloat_FromDouble
from cpython.slice cimport PySlice_GetIndicesEx
from cython cimport freelist
from soket.backend cimport (_default_device, _is_gpu_available,
    Device, DeviceType)
from soket.dtype cimport (_default_datatype, _get_scalar_dtype, promote_types,
    int32, _bool)
from soket.tensor.ops cimport *
from soket.tensor.ops.intern cimport (_copy, _equal, _not_equal, _greater,
    _greater_equal, _less, _less_equal, _argmax, _argmin, _array)
from soket.tensor.creation cimport _ones_like
from soket.autodiff cimport _compute_gradient
from libc.string cimport memcpy, memcmp
from libc.stdlib cimport realloc, calloc, free
cimport cupy as cp
import numpy as np


# Lazy evaluation
cdef bint _LAZY_STATE = False
cdef class LazyState:
    # To store previous lazy evaluation state
    cdef bint _previous_lazy_state

    def __enter__(self) -> None:
        ''' When entering a `with` block. '''

        global _LAZY_STATE
        self._previous_lazy_state = _LAZY_STATE
        _LAZY_STATE = True

    def __exit__(self, *args):
        ''' When exiting a `with` block. '''

        global _LAZY_STATE
        _LAZY_STATE = self._previous_lazy_state

        # LazyState does not handle any exception
        return False

# ROUTINE TO CONTROL LAZY MODE EVALUATION
cpdef lazy(enabled=None):
    if enabled is None:
        return LazyState()

    global _LAZY_STATE
    _LAZY_STATE = <bint> enabled


## SOKET TENSOR ITERATOR ##

cdef class TensorIterator:
    ''' Tensor iterator to iterate over first tensor dimension. '''

    cdef Tensor _x
    cdef int _idx

    def __init__(self, Tensor X):
        self._x = X
        self._idx = 0

    def __iter__(self):
        ''' Prepare the iterator. '''

        self._idx = 0
        return self

    def __next__(self):
        ''' Get next item in the iterator. '''

        if self._idx >= self._x._shape[0]:
            raise StopIteration()

        cdef object value = self._x._getitem(PyLong_FromLong(self._idx))
        self._idx += 1

        return value

## SOKET TENSOR ITERATOR END ##


## OPS ##

# ADDITION
cdef Op _op_elemwise_add = Op(
    _elemwise_add_fwd,
    _elemwise_add_bwd,
    OpType.ELEMWISE_ADD
)

cdef Op _op_scalar_add = Op(
    _scalar_add_fwd,
    _scalar_add_bwd,
    OpType.SCALAR_ADD
)


# NEGATE
cdef Op _op_negate = Op(
    _negate_fwd,
    _negate_bwd,
    OpType.NEGATE
)


# SUBTRACT
cdef Op _op_elemwise_sub = Op(
    _elemwise_sub_fwd,
    _elemwise_sub_bwd,
    OpType.ELEMWISE_SUB
)

cdef Op _op_scalar_sub = Op(
    _scalar_sub_fwd,
    _scalar_sub_bwd,
    OpType.SCALAR_SUB
)


# MULTIPLICATION
cdef Op _op_elemwise_mul = Op(
    _elemwise_mul_fwd,
    _elemwise_mul_bwd,
    OpType.ELEMWISE_MUL
)

cdef Op _op_scalar_mul = Op(
    _scalar_mul_fwd,
    _scalar_mul_bwd,
    OpType.SCALAR_MUL
)


# DIVIDE
cdef Op _op_elemwise_div = Op(
    _elemwise_div_fwd,
    _elemwise_div_bwd,
    OpType.ELEMWISE_DIV
)

cdef Op _op_scalar_div = Op(
    _scalar_div_fwd,
    _scalar_div_bwd,
    OpType.SCALAR_DIV
)


# POWER
cdef Op _op_elemwise_pow = Op(
    _elemwise_pow_fwd,
    _elemwise_pow_bwd,
    OpType.ELEMWISE_POW
)

cdef Op _op_scalar_pow = Op(
    _scalar_pow_fwd,
    _scalar_pow_bwd,
    OpType.SCALAR_POW
)


# BROADCAST
cdef Op _op_broadcast_to = Op(
    _broadcast_to_fwd,
    _broadcast_to_bwd,
    OpType.BROADCAST_TO
)


# SUMMATION
cdef Op _op_sum = Op(
    _sum_fwd,
    _sum_bwd,
    OpType.SUMMATION
)


# ABS
cdef Op _op_abs = Op(
    _abs_fwd,
    _abs_bwd,
    OpType.ABS
)


# MEAN
cdef Op _op_mean = Op(
    _mean_fwd,
    _mean_bwd,
    OpType.MEAN
)


# STANDARD DEVIATION
cdef Op _op_std = Op(
    _std_fwd,
    _std_bwd,
    OpType.STD
)


# VARIANCE
cdef Op _op_var = Op(
    _var_fwd,
    _var_bwd,
    OpType.VAR
)


# MAX and MIN
cdef Op _op_max = Op(
    _max_fwd,
    _max_bwd,
    OpType.MAX
)
cdef Op _op_min = Op(
    _min_fwd,
    _min_bwd,
    OpType.MIN
)


# MATMUL
cdef Op _op_matmul = Op(
    _matmul_fwd,
    _matmul_bwd,
    OpType.MATMUL
)


# RESHAPE
cdef Op _op_reshape = Op(
    _reshape_fwd,
    _reshape_bwd,
    OpType.RESHAPE
)


# PERMUTE
cdef Op _op_permute = Op(
    _permute_fwd,
    _permute_bwd,
    OpType.PERMUTE
)


# TRANSPOSE
cdef Op _op_transpose = Op(
    _transpose_fwd,
    _transpose_bwd,
    OpType.TRANSPOSE
)


# SELECT
cdef Op _op_select = Op(
    _select_fwd,
    _select_bwd,
    OpType.SELECT
)

## OPS END ##


## HELPER DATA-STRUCTURES AND FUNCTIONS ##

# Caching frequently called functions
cdef object cp_ndarray = None
cdef object cp_asnumpy = None

# Import CuPy if GPU support is available.
if _is_gpu_available():
    import cupy as cp

    # Caching
    cp_ndarray = cp.ndarray
    cp_asnumpy = cp.asnumpy


# Supported tensor input types
DEF _SUPPORTED_TYPES_LEN = 5
cdef (PyTypeObject *)[_SUPPORTED_TYPES_LEN] _supported_types = [
    <PyTypeObject *> list,
    <PyTypeObject *> tuple,
    <PyTypeObject *> int,
    <PyTypeObject *> float,
    <PyTypeObject *> bool,
]


cdef inline _ShapeInfo _get_broadcasted_shape_general(
    int *x_shape,
    int x_nshape,

    int *y_shape,
    int y_nshape,

    int nignore  # How many trailing dimensions to ignore
):
    '''
    Get resulting broadcasted shape from two shapes.

    Also supports matmul operation broadcasting.
    '''

    # Python scope forward declaration
    cdef int mas
    cdef int mis

    # Sort shapes in terms of length
    cdef int *max_shape
    cdef int max_nshape
    cdef int *min_shape
    cdef int min_nshape
    if x_nshape >= y_nshape:
        max_shape = x_shape
        max_nshape = x_nshape
        min_shape = y_shape
        min_nshape = y_nshape
    else:
        max_shape = y_shape
        max_nshape = y_nshape
        min_shape = x_shape
        min_nshape = x_nshape

    cdef int diff = max_nshape - min_nshape

    cdef _ShapeInfo info
    info.shape = <int *> malloc(max_nshape * sizeof(int))
    if info.shape == NULL:
        raise MemoryError(f'Could not allocate memory for shape!')

    info.nshape = max_nshape

    # Smaller length shapes filled with 1 at the beginning
    cdef Py_ssize_t i
    for i in range(diff):
        info.shape[i] = max_shape[i]

    for i in range(min_nshape - nignore):
        mis = min_shape[i]
        mas = max_shape[i + diff]

        if mis != 1 and mas != 1 and mis != mas:
            raise RuntimeError(f'Incompatible shapes!')

        info.shape[i + diff] = mas if mas > mis else mis

    return info


cdef inline _ShapeInfo _get_broadcasted_shape(
    int *x_shape,
    int x_nshape,

    int *y_shape,
    int y_nshape
):
    ''' Get resulting broadcasted shape from two shapes. '''

    return _get_broadcasted_shape_general(
        x_shape, x_nshape,
        y_shape, y_nshape,
        0
    )

cdef inline _ShapeInfo _get_matmul_broadcasted_shape(
    int *x_shape,
    int x_nshape,

    int *y_shape,
    int y_nshape
):
    ''' Get resulting broadcasting shape for matmul operation. '''

    cdef _ShapeInfo info = _get_broadcasted_shape_general(
        x_shape, x_nshape,
        y_shape, y_nshape,
        2
    )

    # Resulting matrix dimension
    info.shape[info.nshape - 1] = y_shape[y_nshape - 1]
    info.shape[info.nshape - 2] = x_shape[x_nshape - 2]

    return info

cdef int _get_slice_length(slice slice, Py_ssize_t dim) except -1:
    ''' Get slice length along a given dimension. '''

    cdef Py_ssize_t start, stop, step, slicelength
    if(PySlice_GetIndicesEx(
        slice, dim,
        &start, &stop, &step, &slicelength
    ) < 0):
        raise ValueError('Invalid slice!')

    return <int> slicelength

cdef (int, int) _process_tensor_slice(
    int *shape,
    int nshape,
    int capacity,
    Tensor ten
):
    ''' Processes a tensor slice. '''

    # Check for memory overflow
    cdef int new_nshape = nshape + ten._nshape
    if new_nshape >= capacity:
        capacity = new_nshape * 2
        shape = <int *> realloc(shape, capacity * sizeof(int))

        if shape == NULL:
            raise MemoryError('Cannot reallocate memory to store shape!')

    memcpy(shape + nshape, ten._shape, ten._nshape * sizeof(int))
    return new_nshape, capacity

## HELPER DATA-STRUCTURES AND FUNCTIONS ##


@freelist(512)
cdef class Tensor:
    ''' Represents a soket Tensor. '''

    def __init__(
        self,
        array: object,
        Device device=None,
        DType dtype=None,
        requires_grad=None
    ):
        # Forward declaration (Python scope)
        cdef Tensor other
        cdef _ShapeInfo info
        cdef bint supported = False

        # To be passed in `_tensor_init()`
        cdef object data
        cdef int *shape = NULL
        cdef int nshape = -1
        cdef bint copy_shape = False

        # If given array is Tensor type, copy and create a new tensor
        if Py_TYPE(array) is <PyTypeObject *> Tensor:
            other = <Tensor> array
            device = other._device if device is None else device
            dtype = other._dtype if dtype is None else dtype

            # Compute and get tensor data
            data = other._compute_data()

            # CuPy arrays are not supported in NumPy
            if (device._backend.ndarray is np.ndarray and
            other._device._backend.ndarray is not np.ndarray):
                data = device._backend.array(
                    other._device._backend.asnumpy(data),
                    dtype._str() if dtype is not None else None
                )
            else:
                data = device._backend.array(
                    data,
                    dtype._str() if dtype is not None else None
                )

            # No need to recompute shape
            shape = other._shape
            nshape = other._nshape
            copy_shape = True

        else:
            device = _default_device() if device is None else device

            # NumPy interoperability.
            if Py_TYPE(array) is <PyTypeObject *> np.ndarray:
                data = device._backend.array(
                    array,
                    dtype._str() if dtype is not None else None
                )

            # Availability of GPU indicates CuPy is also available.
            # Hence, interoperability is possible between Soket tensors and CuPy.
            elif (_is_gpu_available() and Py_TYPE(array) is
            <PyTypeObject *> cp_ndarray):
                data = device._backend.array(
                    array if device._backend.ndarray is cp_ndarray
                    else cp_asnumpy(array),
                    dtype._str() if dtype is not None else None
                )

            else:
                # Check if given input object is supported
                for i in range(_SUPPORTED_TYPES_LEN):
                    if Py_TYPE(array) is _supported_types[i]:
                        supported = True
                        break

                if not supported:
                    raise ValueError(f'Unsupported input type: {type(array)}')

                if dtype is None:
                    # For type int, float and bool
                    if i > 1:
                        dtype = _get_scalar_dtype(array)
                    else:
                        dtype = _default_datatype

                if i > 1:
                    shape = <int *> malloc(0)
                    nshape = 0

                data = device._backend.array(array, dtype._str())

            # Deduce datatype if None
            dtype = DType(data.dtype.__str__()) if dtype is None else dtype

            # Extract shapes
            if nshape == -1:
                info = _get_shape_info_from_tuple(data.shape)
                shape = info.shape
                nshape = info.nshape

        self._tensor_init(
            Op(NULL, NULL, OpType.INVALID),
            TensorTriad(NULL, NULL, NULL),
            data,
            device,
            dtype,
            shape, nshape,
            copy_shape,
            requires_grad
        )

    def __dealloc__(self):
        ''' Tensor destructor. '''

        # Release shape memory
        free(self._shape)

        # Input tensors are no longer needed.
        Py_XDECREF(self._inputs.x)
        Py_XDECREF(self._inputs.y)
        Py_XDECREF(self._inputs.Z)

        # Value cache no longer needed.
        if self._value_cache != NULL:
            for i in range(self._n_value_cache):
                Py_XDECREF(self._value_cache[i])

            free(self._value_cache)

    ## DUNDER METHODS ##

    def __setitem__(self,
        idx: int | slice | tuple[int | slice],
        value: Tensor | any
    ) -> None:
        ''' Set element(s) to a tensor. '''

        self._setitem(idx, value)

    def __len__(self) -> int:
        ''' Returns the length of the ndarray. '''

        if self._nshape == 0:  # scalar
            raise TypeError('Attempt to call len() on a scalar tensor')

        return PyLong_FromLong(self._shape[0])

    # Iterator
    def __iter__(self) -> TensorIterator:
        ''' Return a TensorIterator for item iteration. '''

        if self._nshape == 0:
            raise TypeError('Attempt to iterate over a 0D tensor.')

        return TensorIterator(self)

    # Array protocol
    def __array__(self) -> np.ndarray:
        '''
        NumPy array protocol to seamlessly convert to NumPy array, using
        funtions like: np.asanyarray() or np.asarray().
        '''

        return self.numpy()

    def __index__(self) -> int:
        ''' When a tensor is used as a slice index. '''

        cdef int di = self._dtype._idx
        if di < 3 or di > 10:  # from int8 to uint64
            raise TypeError('Only integer tensors can be converted to index!')

        # scalar
        cdef Py_ssize_t size = 1
        for i in range(self._nshape):
            size *= self._shape[i]

        if size != 1:
            raise TypeError(
                'Only tensors with a single element can be converted to index!'
            )

        return self._compute_data().item()

    def __repr__(self) -> str:
        ''' Tensor representation. '''

        cdef str brand = 'soket.Tensor('
        cdef str prefix = '             '

        cdef list lines = str(self._compute_data()).splitlines()
        cdef str res = brand + '\n'.join(
            lines[:1] +
            [prefix + line for line in lines[1:]]
        ) + ', dtype=' + self._dtype._str()

        # Add device information if GPU.
        if self._device._dev_idx == DeviceType.GPU:
            res += ', device=GPU:' + str(self._device.id)

        if self._requires_grad is True:
            res += ', requires_grad=True'

        return res + ')'

    ## DUNDER METHODS END ##

    ## PROPERTIES ##

    @property
    def dtype(self) -> DType:
        ''' Returns datatype of the tensor. '''

        return self._dtype

    @property
    def shape(self) -> tuple:
        ''' Returns shape/dimensions of the tensor. '''

        return self._compute_data().shape

    @property
    def size(self) -> int:
        ''' Returns total number of elements of the tensor. '''

        return self._compute_data().size

    @property
    def device(self) -> Device:
        ''' Returns the device tensor is using. '''

        return self._device

    @property
    def grad(self) -> Tensor:
        ''' Return the gradient of the tensor. '''

        return <Tensor> self._grad

    @property
    def T(self) -> Tensor:
        ''' Get transposed tensor. '''

        return self._transpose()

    @property
    def data(self) -> Tensor:
        ''' Get a detached tensor. '''

        return self._detach()

    @data.setter
    def data(self, Tensor data) -> None:
        ''' Set current tensor object data.'''

        self._set_data(data)

    @property
    def requires_grad(self) -> bool:
        ''' Returns true current tensor requires gradient to be computed. '''

        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self, bint mode) -> None:
        ''' Set requires_grad field but only for leaf/detached tensors. '''

        if self._op.type != OpType.INVALID:
            raise RuntimeError('Only detached/leaf tensors can requires_grad!')

        self._requires_grad = mode

    ## PROPERTIES END ##

    ## METHODS ##

    cpdef Tensor to(self, Device device):
        ''' Migrate a tensor to given device. '''

        if self._device._eq(device):
            return self

        return Tensor(
            self, device, self._dtype,
            True if self._requires_grad else False
        )

    cpdef Tensor copy(self, requires_grad=False):
        '''
        Creates a copy of current tensor.

        But the new tensor is not part of a computational graph.
        '''

        cdef Tensor res = Tensor.__new__(Tensor)
        res._tensor_init(
            self._op,
            TensorTriad(NULL, NULL, NULL),
            _copy(self._device._dev_idx)(self._data_),
            self._device,
            self._dtype,
            self._shape, self._nshape,
            True,
            requires_grad
        )

        return res

    cpdef void backward(self, Tensor adj=None):
        '''
        Computes gradients of all the Nodes upto this node using Reverse Mode
        Automatic Differentiation.

        If no adjoint is provided, default adjoint will be tensor filled with
        ones.
          d(out) / d(out) = 1
        '''

        # Can call backward() only on require grad tensors
        if self._requires_grad == False:
            raise TypeError(
                'Can call backward() only on tensors with requires_grad=True'
            )

        if adj is not None:
            # Device compatibility
            if self._device._ne(adj._device):
                raise RuntimeError('Incompatible adjoint device!')

            # Datatype compatibility
            if self._dtype._idx != adj._dtype._idx:
                raise RuntimeError('Incompatible adjoint datatype!')

            # Shape compatibility
            if (self._nshape != adj._nshape or
            memcmp(self._shape, adj._shape, self._nshape * sizeof(int))):
                raise ValueError('Incompatible adjoint shape!')

        _compute_gradient(
            self,
            _ones_like(
                self,
                self._device,
                self._dtype,
                False
            ) if adj is None else adj
        )

    cpdef Tensor abs(self):
        ''' Return a tensor with absolute values of the elements. '''

        return Tensor._make_from_op(
            _op_abs,
            TensorTriad(<PyObject *> self, NULL, NULL),
            self._device,
            self._dtype,
            self._shape, self._nshape,
            True,
            NULL, 0
        )

    def broadcast_to(self, *shape) -> Tensor:
        ''' Broadcasts tensor to given shape. '''

        return self._broadcast_to(_get_proper_shape(shape))

    def sum(self, *shape, DType dtype=None, keepdims=False) -> Tensor:
        ''' Reduce a tensor performing summation over given axes. '''

        dtype = self._dtype if dtype is None else dtype
        return self._sum(_get_proper_reduction_axes(shape), dtype, keepdims)

    def mean(self, *shape, DType dtype=None, keepdims=False) -> Tensor:
        ''' Reduce a tensor performing mean over given axes. '''

        dtype = self._dtype if dtype is None else dtype
        return self._mean(_get_proper_reduction_axes(shape), dtype, keepdims)

    def std(
        self,
        *shape,
        DType dtype=None,
        keepdims=False,
        correction=1
    ) -> Tensor:
        ''' Computes the standard deviation along the specific axes. '''

        dtype = self._dtype if dtype is None else dtype
        return self._std(
            _get_proper_reduction_axes(shape),
            dtype,
            keepdims,
            correction
        )

    def var(
        self,
        *shape,
        DType dtype=None,
        keepdims=False,
        correction=1
    ) -> Tensor:
        ''' Computes the variance along the specific axes. '''

        dtype = self._dtype if dtype is None else dtype
        return self._var(
            _get_proper_reduction_axes(shape),
            dtype,
            keepdims,
            correction
        )

    def max(self, *shape, DType dtype=None, keepdims=False) -> Tensor:
        ''' Find maximum value(s) in given axes. '''

        dtype = self._dtype if dtype is None else dtype
        return self._max(_get_proper_reduction_axes(shape), dtype, keepdims)

    def min(self, *shape, DType dtype=None, keepdims=False) -> Tensor:
        ''' Find minimum value(s) in given axes. '''

        dtype = self._dtype if dtype is None else dtype
        return self._min(_get_proper_reduction_axes(shape), dtype, keepdims)

    def reshape(self, *shape) -> Tensor:
        ''' Reshape/view current tensor with given shape. '''

        return self._reshape(_get_proper_shape(shape))

    def permute(self, *shape):
        ''' Permute dimensions of a tensor. '''

        return self._permute(_get_proper_reduction_axes(shape))

    def transpose(self):
        ''' Transpose the last two dimensions of tensor. '''

        return self._transpose()

    def argmax(self, axis: int = None, keepdims: bool = False) -> Tensor:
        ''' Returns indicies tensor denoting maximum values over an axis. '''

        return self._argmax(axis, keepdims)

    def argmin(self, axis: int = None, keepdims: bool = False) -> Tensor:
        ''' Returns indicies tensor denoting minimum values over an axis. '''

        return self._argmin(axis, keepdims)

    def detach(self) -> Tensor:
        ''' Get a detached tensor. '''

        return self._detach()

    def item(self) -> any:
        ''' Get scalar tensor element. '''

        # Tensor size
        cdef Py_ssize_t size = 1
        for i in range(self._nshape):
            size *= self._shape[i]

        if size != 1:
            raise ValueError(
                f'Tensor with {size} elements cannot be converted to scalar!'
            )

        return self._compute_data().item()

    cpdef np.ndarray numpy(self):
        ''' Convert tensor data to NumPy and return. '''

        # CuPy (GPU)
        if (_is_gpu_available() and Py_TYPE(self._data_) is <PyTypeObject *>
        cp_ndarray):
            return cp_asnumpy(self._data_)

        # NumPy (CPU)
        return <np.ndarray> _array(0)(self._data_)

    def retain_grad(self) -> None:
        '''
        Forces a non-leaf tensor to store computed gradient in the .grad
        field.
        '''

        self._retain_grad = True

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
    ):
        ''' Properly initializes a tensor. '''

        cdef bint req_grad = False
        if requires_grad is not None:
           req_grad = <bint> requires_grad
        else:
            if inputs.x is not NULL:
                req_grad |= (<Tensor> inputs.x)._requires_grad

            if not req_grad and inputs.y is not NULL:
                req_grad |= (<Tensor> inputs.y)._requires_grad

            if not req_grad and inputs.Z is not NULL:
                req_grad |= (<Tensor> inputs.Z)._requires_grad

        # Allocate and parse shape
        if copy_shape is True:
            self._shape = <int *> malloc(nshape * sizeof(int))
            if self._shape is NULL:
                raise MemoryError(f'Could not allocate memory to store shape!')

            memcpy(self._shape, shape, nshape * sizeof(int))
            self._nshape = nshape
        else:
            self._shape = shape
            self._nshape = nshape

        # Increase input reference count, will be decreased when tensor gets
        # freed.
        Py_XINCREF(inputs.x)
        Py_XINCREF(inputs.y)
        Py_XINCREF(inputs.Z)

        # Init tensor
        self._op = op
        self._inputs = inputs
        self._data_ = data
        self._device = device
        self._dtype = dtype
        self._requires_grad = req_grad

        # By default tensor does not retain grad
        self._retain_grad = False

    cdef object _compute_data(self):
        ''' Resolve computational graph computing current tensor data. '''

        if self._data_ is None:
            self._data_ = self._op.fwd(
                self,

                # Input x
                (<Tensor> self._inputs.x)._compute_data()
                    if self._inputs.x != NULL else None,

                # Input y
                (<Tensor> self._inputs.y)._compute_data()
                    if self._inputs.y != NULL else None,

                # Extra input Z
                (<Tensor> self._inputs.Z)._compute_data()
                    if self._inputs.Z != NULL else None
            )

        return self._data_

    cdef Tensor _argmax(self, axis, keepdims):
        '''
        Returns indicies tensor denoting maximum values over an axis (C ONLY).
        '''

        # Python scope forward declaration
        cdef int axs
        cdef int* shape
        cdef int nshape
        cdef int shape_idx

        if axis is not None:
            axs = <int> axis

            # Check if the given axis underflows/overflows
            if axs < -self._nshape or axs >= self._nshape:
                raise ValueError(f'Give axis out of bounds - {axis}')

            # Adjust axis
            axs = axs + self._nshape if axs < 0 else axs

            # Only reduce over a single axis
            nshape = self._nshape - 1
            shape = <int *> malloc(nshape * sizeof(int))
            if shape == NULL:
                raise MemoryError('Failed to allocate memory for argmax shape!')

            shape_idx = 0
            for i in range(self._nshape):
                if i == axs:
                    continue

                shape[shape_idx] = self._shape[i]
                shape_idx += 1
        else:
            # Scalar tensor
            shape = <int *> malloc(0)
            nshape = 0

        return Tensor._make_const(
            _array(self._device._dev_idx)(
                _argmax(self._device._dev_idx)(
                    self._compute_data(),
                    axis,
                    keepdims=keepdims
                ),
                'int32'  # dtype
            ),
            self._device,
            int32,
            shape, nshape,
            False,
            False
        )

    cdef Tensor _argmin(self, axis, keepdims):
        '''
        Returns indicies tensor denoting minimum values over an axis (C ONLY).
        '''

        # Python scope forward declaration
        cdef int axs
        cdef int* shape
        cdef int nshape
        cdef int shape_idx

        if axis is not None:
            axs = <int> axis

            # Check if the given axis underflows/overflows
            if axs < -self._nshape or axs >= self._nshape:
                raise ValueError(f'Give axis out of bounds - {axis}')

            # Adjust axis
            axs = axs + self._nshape if axs < 0 else axs

            # Only reduce over a single axis
            nshape = self._nshape - 1
            shape = <int *> malloc(nshape * sizeof(int))
            if shape == NULL:
                raise MemoryError('Failed to allocate memory for argmax shape!')

            shape_idx = 0
            for i in range(self._nshape):
                if i == axs:
                    continue

                shape[shape_idx] = self._shape[i]
                shape_idx += 1
        else:
            # Scalar tensor
            shape = <int *> malloc(0)
            nshape = 0

        return Tensor._make_const(
            _array(self._device._dev_idx)(
                _argmin(self._device._dev_idx)(
                    self._compute_data(),
                    axis,
                    keepdims=keepdims
                ),
                'int32'  # dtype
            ),
            self._device,
            int32,
            shape, nshape,
            False,
            False
        )

    cdef void _setitem(
        self,
        idx: int | slice | tuple[int | slice],
        value: Tensor | any
    ):
        ''' Set element(s) to a tensor (C ONLY). '''

        # Python forward declaration
        cdef bint found

        if self._requires_grad:
            raise RuntimeError('Cannot set value to a parameter!')

        cdef object data
        if Py_TYPE(value) is <PyTypeObject *> Tensor:
            value = (<Tensor> value)._compute_data()
        else:
            found = False
            for st in _supported_types[2:_SUPPORTED_TYPES_LEN]:
                if Py_TYPE(value) is st:
                    found = True
                    break

            if not found:
                raise RuntimeError(f'Unsupported item - {value}')

        self._compute_data().__setitem__(idx, value)

    cdef Tensor _detach(self):
        ''' Get detached tensor (C ONLY). '''

        return Tensor._make_const(
            self._compute_data(),
            self._device,
            self._dtype,
            self._shape, self._nshape,
            True,
            False
        )

    cdef Tensor _data(self):
        ''' Alias to Tensor._detach(). '''

        return self._detach()

    cdef void _set_data(self, Tensor data):
        ''' Set tensor object data. '''

        # If `data` is of different shape, change shape
        if (self._nshape != data._nshape or
        memcmp(self._shape, data._shape, self._nshape * sizeof(int))):
            free(self._shape)
            self._shape = <int *> malloc(data._nshape * sizeof(int))
            memcpy(self._shape, data._shape, data._nshape * sizeof(int))

            self._nshape = data._nshape

        self._data_ = data._compute_data()
        self._device = data._device
        self._dtype = data._dtype

    ## CDEF METHODS END ##

    ## CDEF STATIC METHODS ##

    @staticmethod
    cdef Tensor _make_const(
        data,
        Device device,
        DType dtype,
        int *shape, int nshape,
        bint copy_shape,
        requires_grad
    ):
        '''
        Generic construction function for creating detached tensors.

        This routine does not check for valid device, use with caution.
        '''

        # Create Tensor object w/o calling constructor
        cdef Tensor res = Tensor.__new__(Tensor)
        res._tensor_init(
            Op(NULL, NULL, OpType.INVALID),
            TensorTriad(NULL, NULL, NULL),
            data,
            device,
            dtype,
            shape, nshape,
            copy_shape,
            requires_grad
        )

        return res

    @staticmethod
    cdef Tensor _make_from_op(
        Op op,
        TensorTriad inputs,
        Device device,
        DType dtype,
        int *shape, int nshape,
        bint copy_shape,
        PyObject **cache, int ncache
    ):
        ''' Constructs a new tensor from an operation. '''

        # Create Tensor object w/o calling the constructor
        cdef Tensor res = Tensor.__new__(Tensor)
        res._tensor_init(
            op,
            inputs,
            None,
            device,
            dtype,
            shape, nshape,
            copy_shape,
            None
        )

        # Update value caches.
        if ncache > 0:
            res._value_cache = <PyObject **> malloc(ncache * sizeof(PyObject *))
            if res._value_cache == NULL:
                raise MemoryError(
                    'Failed to allocate memory to store value caches!'
                )

            res._n_value_cache = ncache if ncache > 0 else 0
            for i in range(ncache):
                Py_XINCREF(cache[i])
                res._value_cache[i] = cache[i]

        # Compute the operation if not in lazy state.
        if not _LAZY_STATE:
            res._compute_data()

            # Calling `res._tensor_init()` builds a computational graph by
            # filling in operator structures. If gradient calculation is not
            # required, no need to build a computational graph.
            if not res._requires_grad:
                res._op = Op(NULL, NULL, OpType.INVALID)

                Py_XDECREF(res._inputs.x)
                Py_XDECREF(res._inputs.y)
                Py_XDECREF(res._inputs.Z)
                res._inputs = TensorTriad(NULL, NULL, NULL)

        return res

    @staticmethod
    cdef Tensor _from_numpy(object array):
        ''' Construct a tensor from NumPy array (C ONLY). '''

        cdef _ShapeInfo info = _get_shape_info_from_tuple(array.shape)
        return Tensor._make_const(
            array,
            Device(DeviceType.CPU, -1),
            DType(array.dtype.__str__()),
            info.shape, info.nshape,
            False,
            False
        )


    ## CDEF STATIC METHODS END ##

    ## STATIC METHODS ##

    @staticmethod
    def from_numpy(object array) -> Tensor:
        ''' Construct a tensor from NumPy array. '''

        return Tensor._from_numpy(array)

    ## STATIC METHODS END ##

    ## CDEF OPERATOR METHODS ##

    cdef Tensor _add(self, other: Tensor | any):
        ''' Performs addition with a tensor or a scalar (C ONLY). '''

        # Python scope forward declarations
        cdef Tensor ten
        cdef DType dtype
        cdef _ShapeInfo info

        # Initialize first input
        cdef TensorTriad inputs = TensorTriad(<PyObject *> self, NULL, NULL)

        # With another tensor
        if Py_TYPE(other) is <PyTypeObject *> Tensor:
            ten = <Tensor> other
            inputs.y = <PyObject *> other

            # Check if devices are compatible.
            if self._device._ne(ten._device):
                raise RuntimeError(
                    f'Incompatible devices - {self._device} and {ten._device}'
                )

            if self._dtype._idx == ten._dtype._idx:
                dtype = self._dtype
            else:
                dtype = promote_types(self._dtype, ten._dtype)

            info = _get_broadcasted_shape(
                self._shape,
                self._nshape,

                ten._shape,
                ten._nshape
            )

            return Tensor._make_from_op(
                _op_elemwise_add,
                inputs,
                self._device,
                dtype,
                info.shape, info.nshape,
                False,
                NULL, 0
            )

        # With another scalar
        cdef bint found = False
        for st in _supported_types[2:_SUPPORTED_TYPES_LEN]:
            if Py_TYPE(other) is st:
                found = True
                break

        if not found:
            raise RuntimeError(f'Usupported scalar - {other}')

        # Deduce scalar datatype
        dtype = _get_scalar_dtype(other)
        if self._dtype._idx != dtype._idx:
            dtype = promote_types(self._dtype, dtype)

        # Value cache
        cdef (PyObject *)[1] value_cache = [ <PyObject *> other ]

        return Tensor._make_from_op(
            _op_scalar_add,
            inputs,
            self._device,
            dtype,
            self._shape, self._nshape,
            True,
            value_cache, 1
        )

    cdef Tensor _neg(self):
        ''' Negates a tensor (C ONLY). '''

        return Tensor._make_from_op(
            _op_negate,
            TensorTriad(<PyObject *> self, NULL, NULL),
            self._device,
            self._dtype,
            self._shape, self._nshape,
            True,
            NULL, 0
        )

    cdef Tensor _sub(self, other: Tensor | any):
        ''' Subtracts another tensor or a scalar from self tensor (C ONLY). '''

        # Python scope forward declaration
        cdef Tensor ten
        cdef DType dtype
        cdef _ShapeInfo info

        # Initialize first input
        cdef TensorTriad inputs = TensorTriad(<PyObject *> self, NULL, NULL)

        # With another tensor
        if Py_TYPE(other) is <PyTypeObject *> Tensor:
            ten = <Tensor> other
            inputs.y = <PyObject *> other

            # Check if devices are compatible.
            if self._device._ne(ten._device):
                raise RuntimeError(
                    f'Incompatible devices - {self._device} and {ten._device}'
                )

            if self._dtype._idx == ten._dtype._idx:
                dtype = self._dtype
            else:
                dtype = promote_types(self._dtype, ten._dtype)

            info = _get_broadcasted_shape(
                self._shape,
                self._nshape,

                ten._shape,
                ten._nshape
            )

            return Tensor._make_from_op(
                _op_elemwise_sub,
                inputs,
                self._device,
                dtype,
                info.shape, info.nshape,
                False,
                NULL, 0
            )

        # With another scalar
        cdef bint found = False
        for st in _supported_types[2:_SUPPORTED_TYPES_LEN]:
            if Py_TYPE(other) is st:
                found = True
                break

        if not found:
            raise RuntimeError(f'Usupported scalar - {other}')

        # Deduce scalar datatype
        dtype = _get_scalar_dtype(other)
        if self._dtype._idx != dtype._idx:
            dtype = promote_types(self._dtype, dtype)

        # Value cache
        cdef (PyObject *)[2] value_cache = [
            <PyObject *> other,
            <PyObject *> 0x00  # Don't commute
        ]

        return Tensor._make_from_op(
            _op_scalar_sub,
            inputs,
            self._device,
            dtype,
            self._shape, self._nshape,
            True,
            value_cache, 2
        )

    cdef Tensor _rsub(self, other: Tensor | any):
        ''' Subtracts self tensor from a scalar (C ONLY). '''

        # Python scope forward declaration
        cdef Tensor ten
        cdef DType dtype
        cdef _ShapeInfo info

        # Initialize first input
        cdef TensorTriad inputs = TensorTriad(NULL, <PyObject *> self, NULL)

        # With another tensor
        if Py_TYPE(other) is <PyTypeObject *> Tensor:
            raise RuntimeError(
                'Tensor._rsub() is only for scalar subtraction only. Use '
                'Tensor._sub() for element-wise subtraction.'
            )

        # With another scalar
        cdef bint found = False
        for st in _supported_types[2:_SUPPORTED_TYPES_LEN]:
            if Py_TYPE(other) is st:
                found = True
                break

        if not found:
            raise RuntimeError(f'Usupported scalar - {other}')

        # Deduce scalar datatype
        dtype = _get_scalar_dtype(other)
        if self._dtype._idx != dtype._idx:
            dtype = promote_types(self._dtype, dtype)

        # Value cache
        cdef (PyObject *)[2] value_cache = [
            <PyObject *> 0x00,  # Commute
            <PyObject *> other
        ]

        return Tensor._make_from_op(
            _op_scalar_sub,
            inputs,
            self._device,
            dtype,
            self._shape, self._nshape,
            True,
            value_cache, 2
        )

    cdef Tensor _mul(self, other: Tensor | any):
        ''' Performs multiplication with a tensor or a scalar (C ONLY). '''

        # Python scope forward declaration
        cdef Tensor ten
        cdef DType dtype
        cdef _ShapeInfo info

        # Initialize first input
        cdef TensorTriad inputs = TensorTriad(<PyObject *> self, NULL, NULL)

        # With another tensor
        if Py_TYPE(other) is <PyTypeObject *> Tensor:
            ten = <Tensor> other
            inputs.y = <PyObject *> other

            # Check if devices are compatible.
            if self._device._ne(ten._device):
                raise RuntimeError(
                    f'Incompatible devices - {self._device} and {ten._device}'
                )

            if self._dtype._idx == ten._dtype._idx:
                dtype = self._dtype
            else:
                dtype = promote_types(self._dtype, ten._dtype)

            info = _get_broadcasted_shape(
                self._shape,
                self._nshape,

                ten._shape,
                ten._nshape
            )

            return Tensor._make_from_op(
                _op_elemwise_mul,
                inputs,
                self._device,
                dtype,
                info.shape, info.nshape,
                False,
                NULL, 0
            )

        # With another scalar
        cdef bint found = False
        for st in _supported_types[2:_SUPPORTED_TYPES_LEN]:
            if Py_TYPE(other) is st:
                found = True
                break

        if not found:
            raise RuntimeError(f'Usupported scalar - {other}')

        # Deduce scalar datatype
        dtype = _get_scalar_dtype(other)
        if self._dtype._idx != dtype._idx:
            dtype = promote_types(self._dtype, dtype)

        # Value cache
        cdef (PyObject *)[1] value_cache = [ <PyObject *> other ]

        return Tensor._make_from_op(
            _op_scalar_mul,
            inputs,
            self._device,
            dtype,
            self._shape, self._nshape,
            True,
            value_cache, 1
        )

    cdef Tensor _div(self, other: Tensor | any):
        ''' Divides self tensor by a scalar or another tensor (C ONLY). '''

        # Python scope forward declaration
        cdef Tensor ten
        cdef DType dtype
        cdef _ShapeInfo info

        # Initialize first input
        cdef TensorTriad inputs = TensorTriad(<PyObject *> self, NULL, NULL)

        # With another tensor
        if Py_TYPE(other) is <PyTypeObject *> Tensor:
            ten = <Tensor> other
            inputs.y = <PyObject *> other

            # Check if devices are compatible.
            if self._device._ne(ten._device):
                raise RuntimeError(
                    f'Incompatible devices - {self._device} and {ten._device}'
                )

            if self._dtype._idx == ten._dtype._idx:
                dtype = self._dtype
            else:
                dtype = promote_types(self._dtype, ten._dtype)

            info = _get_broadcasted_shape(
                self._shape,
                self._nshape,

                ten._shape,
                ten._nshape
            )

            return Tensor._make_from_op(
                _op_elemwise_div,
                inputs,
                self._device,
                dtype,
                info.shape, info.nshape,
                False,
                NULL, 0
            )

        # With another scalar
        cdef bint found = False
        for st in _supported_types[2:_SUPPORTED_TYPES_LEN]:
            if Py_TYPE(other) is st:
                found = True
                break

        if not found:
            raise RuntimeError(f'Usupported scalar - {other}')

        # Deduce scalar datatype
        dtype = _get_scalar_dtype(other)
        if self._dtype._idx != dtype._idx:
            dtype = promote_types(self._dtype, dtype)

        # Value cache
        cdef (PyObject *)[2] value_cache = [
            <PyObject *> other,
            <PyObject *> 0x00  # Don't commute
        ]

        return Tensor._make_from_op(
            _op_scalar_div,
            inputs,
            self._device,
            dtype,
            self._shape, self._nshape,
            True,
            value_cache, 2
        )

    cdef Tensor _rdiv(self, other: Tensor | any):
        ''' Divides a scalar over self tensor (C ONLY). '''

        # Python scope forward declaration
        cdef Tensor ten
        cdef DType dtype
        cdef _ShapeInfo info

        # With another tensor
        if Py_TYPE(other) is <PyTypeObject *> Tensor:
            raise RuntimeError(
                'Tensor._rsub() is only for scalar subtraction only. Use '
                'Tensor._sub() for element-wise subtraction.'
            )

        # With another scalar
        cdef bint found = False
        for st in _supported_types[2:_SUPPORTED_TYPES_LEN]:
            if Py_TYPE(other) is st:
                found = True
                break

        if not found:
            raise RuntimeError(f'Usupported scalar - {other}')

        # Deduce scalar datatype
        dtype = _get_scalar_dtype(other)
        if self._dtype._idx != dtype._idx:
            dtype = promote_types(self._dtype, dtype)

        # Value cache
        cdef (PyObject *)[2] value_cache = [
            <PyObject *> 0x00,  # Commute
            <PyObject *> other
        ]

        return Tensor._make_from_op(
            _op_scalar_div,
            TensorTriad(NULL, <PyObject *> self, NULL),
            self._device,
            dtype,
            self._shape, self._nshape,
            True,
            value_cache, 2
        )

    cdef Tensor _pow(self, other: Tensor | any):
        ''' Raise a tensor to a power (C ONLY). '''

        # Python scope forward declaration
        cdef Tensor ten
        cdef DType dtype
        cdef _ShapeInfo info

        # Initialize first input
        cdef TensorTriad inputs = TensorTriad(<PyObject *> self, NULL, NULL)

        # With another tensor
        if Py_TYPE(other) is <PyTypeObject *> Tensor:
            ten = <Tensor> other
            inputs.y = <PyObject *> other

            # Check if devices are compatible.
            if self._device._ne(ten._device):
                raise RuntimeError(
                    f'Incompatible devices - {self._device} and {ten._device}'
                )

            if self._dtype._idx == ten._dtype._idx:
                dtype = self._dtype
            else:
                dtype = promote_types(self._dtype, ten._dtype)

            info = _get_broadcasted_shape(
                self._shape,
                self._nshape,

                ten._shape,
                ten._nshape
            )

            return Tensor._make_from_op(
                _op_elemwise_pow,
                inputs,
                self._device,
                dtype,
                info.shape, info.nshape,
                False,
                NULL, 0
            )

        # With another scalar
        cdef bint found = False
        for st in _supported_types[2:_SUPPORTED_TYPES_LEN]:
            if Py_TYPE(other) is st:
                found = True
                break

        if not found:
            raise RuntimeError(f'Usupported scalar - {other}')

        # Deduce scalar datatype
        dtype = _get_scalar_dtype(other)
        if self._dtype._idx != dtype._idx:
            dtype = promote_types(self._dtype, dtype)

        # Value cache
        cdef (PyObject *)[2] value_cache = [
            <PyObject *> other,
            <PyObject *> 0x00  # Don't commute
        ]

        return Tensor._make_from_op(
            _op_scalar_pow,
            inputs,
            self._device,
            dtype,
            self._shape, self._nshape,
            True,
            value_cache, 2
        )

    cdef Tensor _rpow(self, other: Tensor | any):
        ''' Raise a scalar to a power (C ONLY). '''

        # Python scope forward declaration
        cdef Tensor ten
        cdef DType dtype
        cdef _ShapeInfo info

        # With another tensor
        if Py_TYPE(other) is <PyTypeObject *> Tensor:
            raise RuntimeError(
                'Tensor._rsub() is only for scalar subtraction only. Use '
                'Tensor._sub() for element-wise subtraction.'
            )

        # With another scalar
        cdef bint found = False
        for st in _supported_types[2:_SUPPORTED_TYPES_LEN]:
            if Py_TYPE(other) is st:
                found = True
                break

        if not found:
            raise RuntimeError(f'Usupported scalar - {other}')

        # Deduce scalar datatype
        dtype = _get_scalar_dtype(other)
        if self._dtype._idx != dtype._idx:
            dtype = promote_types(self._dtype, dtype)

        # Value cache
        cdef (PyObject *)[2] value_cache = [
            <PyObject *> 0x00,  # Commute
            <PyObject *> other
        ]

        return Tensor._make_from_op(
            _op_scalar_pow,
            TensorTriad(NULL, <PyObject *> self, NULL),
            self._device,
            dtype,
            self._shape, self._nshape,
            True,
            value_cache, 2
        )

    cdef Tensor _broadcast_to(self, tuple shape):
        ''' Broadcast tensor to given shape. '''

        cdef int nshape = PyTuple_GET_SIZE(shape)
        cdef int diff = nshape - self._nshape
        cdef int *in_shape = <int *> malloc(nshape * sizeof(int))
        cdef object elem

        if diff < 0:
            free(in_shape)
            raise RuntimeError(
                'Attempt to reduce a tensor using broadcast_to()'
            )

        cdef Py_ssize_t i
        for i in range(diff):
            in_shape[i] = <int> <object> PyTuple_GET_ITEM(shape, i)

        cdef int s
        cdef int self_s
        for i in range(nshape):
            elem = <object> PyTuple_GET_ITEM(shape, i)

            if Py_TYPE(elem) is not <PyTypeObject *> int:
                free(in_shape)
                raise TypeError('Expected broadcast shape to be integers!')

            s = <int> elem
            in_shape[i] = s

            ## NOTE: This difference helps account for scalar tensors as well.
            if i < diff:
                continue

            self_s = self._shape[i - diff]
            if self_s != s and self_s != 1:
                free(in_shape)
                raise RuntimeError('Incompatible broadcast shapes!')

        cdef (PyObject *)[1] value_cache = [ <PyObject *> shape ]

        return Tensor._make_from_op(
            _op_broadcast_to,
            TensorTriad(<PyObject *> self, NULL, NULL),
            self._device,
            self._dtype,
            in_shape, nshape,
            False,
            value_cache, 1
        )

    cdef Tensor _sum(self, tuple axes, DType dtype, keepdims):
        ''' Performs summation reduction on tensors (C ONLY). '''

        cdef int *shape
        cdef int nshape
        shape, nshape = _validate_and_create_reduction_axes(
            axes,
            self._shape, self._nshape,
            keepdims is True
        )

        # Sum over all axis
        if nshape == 0:
            axes = None

        # Value cache
        cdef (PyObject *)[2] value_cache = [
            <PyObject *> axes,
            <PyObject *> keepdims
        ]

        return Tensor._make_from_op(
            _op_sum,
            TensorTriad(<PyObject *> self, NULL, NULL),
            self._device,
            dtype,
            shape, nshape,
            False,
            value_cache, 2
        )

    cdef Tensor _mean(self, tuple axes, DType dtype, keepdims):
        ''' Performs mean reduction on tensors (C ONLY). '''

        cdef int *shape
        cdef int nshape
        cdef int size

        shape, nshape = _validate_and_create_reduction_axes(
            axes,
            self._shape, self._nshape,
            keepdims is True
        )

        # Find the number of observations
        cdef long observations = 1
        if self._nshape == nshape:  # keepdims=True
            for i in range(nshape):
                if self._shape[i] != shape[i]:
                    observations *= self._shape[i]
        else:
            if nshape == 0:  # scalar
                for i in range(self._nshape):
                    observations *= self._shape[i]
            else:
                size = PyTuple_GET_SIZE(axes)
                for i in range(size):
                    observations *= self._shape[
                        PyLong_AsLong(<object> PyTuple_GET_ITEM(axes, i))
                    ]

        cdef object py_observations = PyLong_FromLong(observations)

        # Mean over all axis
        if nshape == 0:
            axes = None

        # Value cache
        cdef (PyObject *)[3] value_cache = [
            <PyObject *> axes,
            <PyObject *> keepdims,
            <PyObject *> py_observations
        ]

        return Tensor._make_from_op(
            _op_mean,
            TensorTriad(<PyObject *> self, NULL, NULL),
            self._device,
            dtype,
            shape, nshape,
            False,
            value_cache, 3
        )

    cdef Tensor _std(self, tuple axes, DType dtype, keepdims, correction):
        ''' Computes the standard deviation of tensor elements (C ONLY). '''

        cdef int *shape
        cdef int nshape
        cdef int size

        shape, nshape = _validate_and_create_reduction_axes(
            axes,
            self._shape, self._nshape,
            keepdims is True
        )

        # Find the number of observations
        cdef long observations = 1
        if self._nshape == nshape:  # keepdims=True
            for i in range(nshape):
                if self._shape[i] != shape[i]:
                    observations *= self._shape[i]
        else:
            if nshape == 0:  # scalar
                for i in range(self._nshape):
                    observations *= self._shape[i]
            else:
                size = PyTuple_GET_SIZE(axes)
                for i in range(size):
                    observations *= self._shape[
                        PyLong_AsLong(<object> PyTuple_GET_ITEM(axes, i))
                    ]

        # Apply correction
        observations -= PyLong_AsLong(correction)
        observations = 0 if observations < 0 else observations
        cdef object py_observations = PyLong_FromLong(observations)

        # Compute standard deviation over all axis
        if nshape == 0:
            axes = None

        # Value cache
        cdef (PyObject *)[5] value_cache = [
            <PyObject *> axes,
            <PyObject *> keepdims,
            <PyObject *> py_observations,
            <PyObject *> None,  # mean
            <PyObject *> correction
        ]

        return Tensor._make_from_op(
            _op_std,
            TensorTriad(<PyObject *> self, NULL, NULL),
            self._device,
            dtype,
            shape, nshape,
            False,
            value_cache, 5
        )

    cdef Tensor _var(self, tuple axes, DType dtype, keepdims, correction):
        ''' Computes the variance of tensor elements over some axes (C ONLY). '''

        cdef int *shape
        cdef int nshape
        cdef int size

        shape, nshape = _validate_and_create_reduction_axes(
            axes,
            self._shape, self._nshape,
            keepdims is True
        )

        # Find the number of observations
        cdef double observations = 1
        if self._nshape == nshape:  # keepdims=True
            for i in range(nshape):
                if self._shape[i] != shape[i]:
                    observations *= self._shape[i]
        else:
            if nshape == 0:  # scalar
                for i in range(self._nshape):
                    observations *= self._shape[i]
            else:
                size = PyTuple_GET_SIZE(axes)
                for i in range(size):
                    observations *= self._shape[
                        PyLong_AsLong(<object> PyTuple_GET_ITEM(axes, i))
                    ]

        # Apply correction
        observations -= PyLong_AsLong(correction)
        observations = 2 / (0 if observations < 0 else observations)
        cdef object py_observations = PyFloat_FromDouble(observations)

        # Compute standard deviation over all axis
        if nshape == 0:
            axes = None

        # Value cache
        cdef (PyObject *)[5] value_cache = [
            <PyObject *> axes,
            <PyObject *> keepdims,
            <PyObject *> py_observations,
            <PyObject *> None,  # mean
            <PyObject *> correction
        ]

        return Tensor._make_from_op(
            _op_var,
            TensorTriad(<PyObject *> self, NULL, NULL),
            self._device,
            dtype,
            shape, nshape,
            False,
            value_cache, 5
        )

    cdef Tensor _max(self, tuple axes, DType dtype, keepdims):
        ''' Find maximum value(s) in given axes (C ONLY). '''

        cdef int *shape
        cdef int nshape

        shape, nshape = _validate_and_create_reduction_axes(
            axes,
            self._shape, self._nshape,
            keepdims is True
        )

        # Find maximum over all axis
        if nshape == 0:
            axes = None

        # Value cache
        cdef (PyObject *)[2] value_cache = [
            <PyObject *> axes,
            <PyObject *> keepdims
        ]

        return Tensor._make_from_op(
            _op_max,
            TensorTriad(<PyObject *> self, NULL, NULL),
            self._device,
            dtype,
            shape, nshape,
            False,
            value_cache, 2
        )

    cdef Tensor _min(self, tuple axes, DType dtype, keepdims):
        ''' Find minimum value(s) in given axes (C ONLY). '''

        cdef int *shape
        cdef int nshape

        shape, nshape = _validate_and_create_reduction_axes(
            axes,
            self._shape, self._nshape,
            keepdims is True
        )

        # Find maximum over all axis
        if nshape == 0:
            axes = None

        # Value cache
        cdef (PyObject *)[2] value_cache = [
            <PyObject *> axes,
            <PyObject *> keepdims
        ]

        return Tensor._make_from_op(
            _op_min,
            TensorTriad(<PyObject *> self, NULL, NULL),
            self._device,
            dtype,
            shape, nshape,
            False,
            value_cache, 2
        )

    cdef Tensor _matmul(self, Tensor other):
        ''' Perform matrix-matrix multiply against another tensor (C ONLY). '''

        # Check if devices are compatible.
        if self._device._ne(other._device):
            raise RuntimeError(
                f'Incompatible devices - {self._device} and {other._device}'
            )

        if self._nshape < 2 or other._nshape < 2:
            raise RuntimeError('Tensor shape has to be atleast 2D for matmul')

        if self._shape[self._nshape - 1] != other._shape[other._nshape - 2]:
            raise RuntimeError('Incompatible shapes for matmul!')

        cdef DType dtype
        if self._dtype._idx == other._dtype._idx:
            dtype = self._dtype
        else:
            dtype = promote_types(self._dtype, other._dtype)

        cdef _ShapeInfo info = _get_matmul_broadcasted_shape(
            self._shape,
            self._nshape,

            other._shape,
            other._nshape
        )

        return Tensor._make_from_op(
            _op_matmul,
            TensorTriad(<PyObject *> self, <PyObject *> other, NULL),
            self._device,
            dtype,
            info.shape, info.nshape,
            False,
            NULL, 0
        )

    cdef Tensor _reshape(self, tuple shape):
        ''' Reshape/view current tensor with given shape (C ONLY). '''

        cdef int self_size = 1
        for i in range(self._nshape):
            self_size *= self._shape[i]

        cdef int new_nshape = PyTuple_GET_SIZE(shape)
        cdef int *new_shape = <int *> malloc(new_nshape * sizeof(int))
        if new_shape == NULL:
            raise MemoryError('Failed to allocate memory or shape!')

        cdef int sh
        cdef int new_shape_size = 1
        cdef int inferred_i = -1
        for i in range(new_nshape):
            sh = <int> <object> PyTuple_GET_ITEM(shape, i)
            if sh < 0:
                if inferred_i != -1:
                    free(new_shape)
                    raise ValueError('Can only infer one dimension!')
                inferred_i = i
            else:
                new_shape_size *= sh
                new_shape[i] = sh

        # Inferred dimension
        cdef int inferred_dim
        if inferred_i != -1:
            inferred_dim = self_size / new_shape_size  # Index variable as shape
            new_shape_size *= inferred_dim

            # Plug in the inferred dimension
            new_shape[inferred_i] = inferred_dim

        if self_size != new_shape_size:
            free(new_shape)
            raise ValueError(f'Cannot reshape {self_size} elements to {shape}!')

        # Caching
        cdef (PyObject *)[1] value_cache = [ <PyObject *> shape ]

        return Tensor._make_from_op(
            _op_reshape,
            TensorTriad(<PyObject *> self, NULL, NULL),
            self._device,
            self._dtype,
            new_shape, new_nshape,
            False,
            value_cache, 1
        )

    cdef Tensor _permute(self, tuple axes):
        ''' Permute axes of the tensor (C ONLY). '''

        cdef int nshape = self._nshape
        cdef int *shape = <int *> malloc(nshape * sizeof(int))
        if shape == NULL:
            raise MemoryError(f'Failed to allocate permute shape!')

        if PyTuple_GET_SIZE(axes) != nshape:
            raise ValueError('The permute axes should cover all tensor axes.')

        cdef int *mask = <int *> calloc(nshape, sizeof(int))
        cdef int axis
        for i in range(nshape):
            axis = <int> <object> PyTuple_GET_ITEM(axes, i)

            if axis < -nshape or axis >= nshape:
                raise ValueError(
                    f'Permutation axis exceeding shape - at index {i}'
                )

            # Adjust axis
            axis = axis + nshape if axis < 0 else axis

            # Check and mark the axis as permuted
            if mask[axis] == 1:
                raise ValueError('Attempt to permute already permuted axis!')

            mask[axis] = 1  # Permuted
            shape[i] = self._shape[axis]

        # There should not be any remaining unpermuted axis
        for i in range(nshape):
            if not mask[i]:
                raise ValueError('Provided axes should cover all dimensions!')

        free(mask)

        # Value cache
        cdef (PyObject *)[1] value_cache = [
            <PyObject *> axes
        ]

        return Tensor._make_from_op(
            _op_permute,
            TensorTriad(<PyObject *> self, NULL, NULL),
            self._device,
            self._dtype,
            shape, nshape,
            False,
            value_cache, 1
        )

    cdef Tensor _transpose(self):
        ''' Get transposed tensor (C ONLY). '''

        # Shape has to be atleast 2 dimensional
        if self._nshape < 2:
            raise RuntimeError('Shape has to be atleast 2D for transpose')

        cdef int nshape = self._nshape
        cdef int *shape = <int *> malloc(nshape * sizeof(int))
        if shape == NULL:
            raise MemoryError(f'Failed to allocate transpose shape!')

        # Transpose entire shape
        cdef Py_ssize_t i
        for i in range(nshape - 1, -1, -1):
            shape[i] = self._shape[nshape - i - 1]

        return Tensor._make_from_op(
            _op_transpose,
            TensorTriad(<PyObject *> self, NULL, NULL),
            self._device,
            self._dtype,
            shape, nshape,
            False,
            NULL, 0
        )

    cdef Tensor _getitem(self, idx: int | slice | tuple[int | slice] | Tensor):
        ''' Get element(s) from given indicies (C ONLY). '''

        # Python scope forward declarations
        cdef int axis
        cdef object index
        cdef Tensor ten_idx

        cdef int *shape = <int *> malloc(self._nshape * sizeof(int))
        if shape == NULL:
            raise MemoryError('Failed to create memory for select shape!')
        cdef int capacity = self._nshape
        cdef int nshape = 0

        cdef Py_ssize_t i = 0
        if Py_TYPE(idx) is <PyTypeObject *> int:
            axis = <int> idx

            if axis < -self._shape[i] or axis >= self._shape[i]:
                free(shape)
                raise ValueError(f'Select index out of bounds - {idx}')

            i += 1
        elif Py_TYPE(idx) is <PyTypeObject *> Tensor:
            ten_idx = <Tensor> idx
            nshape, capacity = _process_tensor_slice(
                shape, nshape, capacity,
                ten_idx
            )

            idx = ten_idx._compute_data()
            i += 1
        elif Py_TYPE(idx) is <PyTypeObject *> slice:
            shape[nshape] = _get_slice_length(<slice> idx, self._shape[i])
            nshape += 1
            i += 1
        elif Py_TYPE(idx) is <PyTypeObject *> tuple:  # tuple
            for i in range(PyTuple_GET_SIZE(<tuple> idx)):
                index = <object> PyTuple_GET_ITEM(idx, i)

                if Py_TYPE(index) is <PyTypeObject *> int:
                    axis = <int> index

                    if axis < -self._shape[i] or axis >= self._shape[i]:
                        free(shape)
                        raise ValueError(
                            f'Select index out of bounds - {index}'
                        )
                elif Py_TYPE(index) is <PyTypeObject *> slice:
                    shape[nshape] = _get_slice_length(
                        <slice> index,
                        self._shape[i]
                    )
                    nshape += 1
                elif Py_TYPE(index) is <PyTypeObject *> Tensor:
                    ten_idx = <Tensor> index
                    nshape, capacity = _process_tensor_slice(
                        shape, nshape, capacity,
                        ten_idx
                    )
                    PyTuple_SET_ITEM(idx, i, <object> ten_idx._compute_data())

            i += 1  # i + 1: Python for loops are unlike C
        else:
            raise ValueError('Unrecognized index type!')

        # Copy remaining shape
        while i < self._nshape:
            shape[nshape] = self._shape[i]
            nshape += 1
            i += 1

        # Value cache
        cdef (PyObject *)[1] value_cache = [
            <PyObject *> idx
        ]

        return Tensor._make_from_op(
            _op_select,
            TensorTriad(<PyObject *> self, NULL, NULL),
            self._device,
            self._dtype,
            shape, nshape,
            False,
            value_cache, 1
        )

    ## CDEF OPERATOR METHODS END ##

    ## CDEF COMPARISON OPERATORS ##

    cdef Tensor _eq(self, other: Tensor | any):
        ''' Test element-wise equality (C ONLY). '''

        # Python scope forward declaration
        cdef Tensor ten
        cdef _ShapeInfo info
        cdef bint copy_shape = True

        if Py_TYPE(other) is <PyTypeObject *> Tensor:
            ten = <Tensor> other

            if self._device._ne(ten._device):
                raise RuntimeError(
                    f'Incompatible devices - {self._device} and {ten._device}'
                )

            other = ten._compute_data()
            info = _get_broadcasted_shape(
                self._shape, self._nshape,
                ten._shape, ten._nshape
            )

            # No need to copy shape.
            copy_shape = False
        else:
            info.shape = self._shape
            info.nshape = self._nshape

        return Tensor._make_const(
            _equal(self._device._dev_idx)(
                self._compute_data(), other
            ),
            self._device,
            _bool,
            info.shape, info.nshape,
            copy_shape,
            False
        )

    cdef Tensor _ne(self, other: Tensor | any):
        ''' Test element-wise inequality (C ONLY). '''

        # Python scope forward declaration
        cdef Tensor ten
        cdef _ShapeInfo info
        cdef bint copy_shape = True

        if Py_TYPE(other) is <PyTypeObject *> Tensor:
            ten = <Tensor> other

            if self._device._ne(ten._device):
                raise RuntimeError(
                    f'Incompatible devices - {self._device} and {ten._device}'
                )

            other = ten._compute_data()
            info = _get_broadcasted_shape(
                self._shape, self._nshape,
                ten._shape, ten._nshape
            )

            # No need to copy shape.
            copy_shape = False
        else:
            info.shape = self._shape
            info.nshape = self._nshape

        return Tensor._make_const(
            _not_equal(self._device._dev_idx)(
                self._compute_data(), other
            ),
            self._device,
            _bool,
            info.shape, info.nshape,
            copy_shape,
            False
        )

    cdef Tensor _gt(self, other: Tensor | any):
        ''' Test element-wise greater (C ONLY). '''

        # Python scope forward declaration
        cdef Tensor ten
        cdef _ShapeInfo info
        cdef bint copy_shape = True

        if Py_TYPE(other) is <PyTypeObject *> Tensor:
            ten = <Tensor> other

            if self._device._ne(ten._device):
                raise RuntimeError(
                    f'Incompatible devices - {self._device} and {ten._device}'
                )

            other = ten._compute_data()
            info = _get_broadcasted_shape(
                self._shape, self._nshape,
                ten._shape, ten._nshape
            )

            # No need to copy shape.
            copy_shape = False
        else:
            info.shape = self._shape
            info.nshape = self._nshape

        return Tensor._make_const(
            _greater(self._device._dev_idx)(
                self._compute_data(), other
            ),
            self._device,
            _bool,
            info.shape, info.nshape,
            copy_shape,
            False
        )

    cdef Tensor _rgt(self, other: Tensor | any):
        ''' Test element-wise greater (RHS and C ONLY). '''

        return Tensor._make_const(
            _greater(self._device._dev_idx)(
                other, self._compute_data()
            ),
            self._device,
            _bool,
            self._shape, self._nshape,
            True,
            False
        )

    cdef Tensor _ge(self, other: Tensor | any):
        ''' Test element-wise greater and equal (C ONLY). '''

        # Python scope forward declaration
        cdef Tensor ten
        cdef _ShapeInfo info
        cdef bint copy_shape = True

        if Py_TYPE(other) is <PyTypeObject *> Tensor:
            ten = <Tensor> other

            if self._device._ne(ten._device):
                raise RuntimeError(
                    f'Incompatible devices - {self._device} and {ten._device}'
                )

            other = ten._compute_data()
            info = _get_broadcasted_shape(
                self._shape, self._nshape,
                ten._shape, ten._nshape
            )

            # No need to copy shape.
            copy_shape = False
        else:
            info.shape = self._shape
            info.nshape = self._nshape

        return Tensor._make_const(
            _greater_equal(self._device._dev_idx)(
                self._compute_data(), other
            ),
            self._device,
            _bool,
            info.shape, info.nshape,
            copy_shape,
            False
        )

    cdef Tensor _rge(self, other: Tensor | any):
        ''' Test element-wise greater (RHS and C ONLY). '''

        return Tensor._make_const(
            _greater_equal(self._device._dev_idx)(
                other, self._compute_data()
            ),
            self._device,
            _bool,
            self._shape, self._nshape,
            True,
            False
        )

    cdef Tensor _lt(self, other: Tensor | any):
        ''' Test element-wise less (C ONLY). '''

        # Python scope forward declaration
        cdef Tensor ten
        cdef _ShapeInfo info
        cdef bint copy_shape = True

        if Py_TYPE(other) is <PyTypeObject *> Tensor:
            ten = <Tensor> other

            if self._device._ne(ten._device):
                raise RuntimeError(
                    f'Incompatible devices - {self._device} and {ten._device}'
                )

            other = ten._compute_data()
            info = _get_broadcasted_shape(
                self._shape, self._nshape,
                ten._shape, ten._nshape
            )

            # No need to copy shape.
            copy_shape = False
        else:
            info.shape = self._shape
            info.nshape = self._nshape

        return Tensor._make_const(
            _less(self._device._dev_idx)(
                self._compute_data(), other
            ),
            self._device,
            _bool,
            info.shape, info.nshape,
            copy_shape,
            False
        )

    cdef Tensor _rlt(self, other: Tensor | any):
        ''' Test element-wise less (RHS and C ONLY). '''

        return Tensor._make_const(
            _less(self._device._dev_idx)(
                other, self._compute_data()
            ),
            self._device,
            _bool,
            self._shape, self._nshape,
            True,
            False
        )

    cdef Tensor _le(self, other: Tensor | any):
        ''' Test element-wise less and equal (C ONLY). '''

        # Python scope forward declaration
        cdef Tensor ten
        cdef _ShapeInfo info
        cdef bint copy_shape = True

        if Py_TYPE(other) is <PyTypeObject *> Tensor:
            ten = <Tensor> other

            if self._device._ne(ten._device):
                raise RuntimeError(
                    f'Incompatible devices - {self._device} and {ten._device}'
                )

            other = ten._compute_data()
            info = _get_broadcasted_shape(
                self._shape, self._nshape,
                ten._shape, ten._nshape
            )

            # No need to copy shape.
            copy_shape = False
        else:
            info.shape = self._shape
            info.nshape = self._nshape

        return Tensor._make_const(
            _less_equal(self._device._dev_idx)(
                self._compute_data(), other
            ),
            self._device,
            _bool,
            info.shape, info.nshape,
            copy_shape,
            False
        )

    cdef Tensor _rle(self, other: Tensor | any):
        ''' Test element-wise less (RHS and C ONLY). '''

        return Tensor._make_const(
            _less_equal(self._device._dev_idx)(
                other, self._compute_data()
            ),
            self._device,
            _bool,
            self._shape, self._nshape,
            True,
            False
        )

    ## CDEF COMPARISON OPERATORS END ##

    ## DUNDER OPERATORS ##

    def __add__(self, other: Tensor | any) -> Tensor:
        ''' Performs addition with another tensor/scalar. '''

        return self._add(other)

    def __radd__(self, other: Tensor | any) -> Tensor:
        ''' Perform Performs addition with another tensor/scalar (RHS). '''

        return self._add(other)

    def __neg__(self) -> Tensor:
        ''' Negates a tensor. '''

        return self._neg()

    def __sub__(self, other: Tensor | any) -> Tensor:
        ''' Subtracts another tensor or a scalar from self tensor. '''

        return self._sub(other)

    def __rsub__(self, other: Tensor | any) -> Tensor:
        ''' Subtracts self tensor from a scalar (RHS). '''

        return self._rsub(other)

    def __mul__(self, other: Tensor | any) -> Tensor:
        ''' Multiplies another tensor or a scalar from self tensor. '''

        return self._mul(other)

    def __rmul__(self, other: Tensor | any) -> Tensor:
        ''' Multiplies another tensor or a scalar from self tensor (RHS). '''

        return self._mul(other)

    def __truediv__(self, other: Tensor | any) -> Tensor:
        ''' Divides self tensor by a scalar or another tensor. '''

        return self._div(other)

    def __rtruediv__(self, other: Tensor | any) -> Tensor:
        ''' Divides a scalar over self tensor (RHS). '''

        return self._rdiv(other)

    def __pow__(self, other: Tensor | any) -> Tensor:
        ''' Raise a tensor to a power '''

        return self._pow(other)

    def __rpow__(self, other: Tensor | any) -> Tensor:
        ''' Raise a scalar to a power (RHS). '''

        return self._rpow(other)

    def __getitem__(self, idx: int | slice | tuple[int | slice]) -> Tensor:
        ''' Get element(s) from given indicies. '''

        return self._getitem(idx)

    ## DUNDER OPERATORS END ##

    ## DUNDER COMPARISON OPERATOR ##

    def __eq__(self, other: Tensor | any) -> Tensor:
        ''' Test element-wise equality. '''

        return self._eq(other)

    def __ne__(self, other: Tensor | any) -> Tensor:
        ''' Test element-wise inequality. '''

        return self._ne(other)

    def __gt__(self, other: Tensor | any) -> Tensor:
        ''' Test element-wise greater. '''

        return self._gt(other)

    def __rgt__(self, other: Tensor | any) -> Tensor:
        ''' Test element-wise greater (RHS). '''

        return self._rgt(other)

    def __ge__(self, other: Tensor | any) -> Tensor:
        ''' Test element-wise greater and equal. '''

        return self._ge(other)

    def __rge__(self, other: Tensor | any) -> Tensor:
        ''' Test element-wise greater equal (RHS). '''

        return self._rge(other)

    def __lt__(self, other: Tensor | any) -> Tensor:
        ''' Test element-wise less. '''

        return self._lt(other)

    def __rlt__(self, other: Tensor | any) -> Tensor:
        ''' Test element-wise less (RHS). '''

        return self._rlt(other)

    def __le__(self, other: Tensor | any) -> Tensor:
        ''' Test element-wise less and equal. '''

        return self._ge(other)

    def __rle__(self, other: Tensor | any) -> Tensor:
        ''' Test element-wise less equal (RHS). '''

        return self._rle(other)

    def __matmul__(self, other: Tensor) -> Tensor:
        ''' Perform matrix-matrix multiply against another tensor. '''

        return self._matmul(other)

    ## DUNDER COMPARISON OPERATOR END ##
