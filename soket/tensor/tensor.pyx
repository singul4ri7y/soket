from cpython.ref cimport Py_XINCREF, Py_XDECREF
from cpython.object cimport Py_TYPE, PyTypeObject
from cpython.tuple cimport PyTuple_GET_SIZE, PyTuple_GET_ITEM
from cpython.long cimport PyLong_FromLong
from cpython.slice cimport PySlice_GetIndicesEx
from cython cimport freelist
from soket.backend cimport _default_device, _is_gpu_available, DeviceType
from soket.dtype cimport (_default_datatype, _get_scalar_dtype, promote_types,
    int32, _bool)
from soket.tensor.ops cimport *
from soket.tensor.ops.intern cimport (_copy, _equal, _not_equal, _greater,
    _greater_equal, _less, _less_equal, _argmax, _argmin, _array)
from soket.tensor.creation cimport _ones_like
from soket.autodiff cimport _compute_gradient
from libc.string cimport memcpy, memcmp
from libc.stdlib cimport calloc, free, qsort
from math import prod
cimport numpy as np
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


# MEAN
cdef Op _op_mean = Op(
    _mean_fwd,
    _mean_bwd,
    OpType.MEAN
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


cdef int _ascending_sort(const void *a, const void *b) noexcept nogil:
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
        if i == _axes[axes_idx]:
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

cdef int _get_slice_length(slice slice, Py_ssize_t dim) except -1:
    ''' Get slice length along a given dimension. '''

    cdef Py_ssize_t start, stop, step, slicelength
    if(PySlice_GetIndicesEx(
        slice, dim,
        &start, &stop, &step, &slicelength
    ) < 0):
        raise ValueError('Invalid slice!')

    return <int> slicelength

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
            OpInput(NULL, NULL),
            data,
            device,
            dtype,
            shape, nshape,
            copy_shape,
            requires_grad
        )

    def __del__(self):
        ''' Tensor destructor. '''

        # Release shape memory
        free(self._shape)

        # Input tensors are no longer needed.
        Py_XDECREF(self._inputs.x)
        Py_XDECREF(self._inputs.y)

        # Value cache no longer needed.
        Py_XDECREF(self._value_cache[0])
        Py_XDECREF(self._value_cache[1])
        Py_XDECREF(self._value_cache[2])
        Py_XDECREF(self._value_cache[3])

    ## DUNDER METHODS ##

    def __setitem__(self,
        idx: int | slice | tuple[int | slice],
        value: Tensor | any
    ) -> None:
        ''' Set element(s) to a tensor. '''

        self._setitem(idx, value)

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
            OpInput(NULL, NULL),
            _copy(self._device._dev_idx)(self._data),
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
            if self._dtype._idx != adj._dtype.idx:
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

    def max(self, *shape, DType dtype=None, keepdims=False) -> Tensor:
        ''' Find maximum value(s) in given axes. '''

        dtype = self._dtype if dtype is None else dtype
        return self._max(_get_proper_reduction_axes(shape), dtype, keepdims)

    def min(self, *shape, DType dtype=None, keepdims=False) -> Tensor:
        ''' Find minimum value(s) in given axes. '''

        dtype = self._dtype if dtype is None else dtype
        return self._min(_get_proper_reduction_axes(shape), dtype, keepdims)

    def argmax(self, axis: int = None, keepdims: bool = False) -> Tensor:
        ''' Returns indicies tensor denoting maximum values over an axis. '''

        return self._argmax(axis, keepdims)

    def argmin(self, axis: int = None, keepdims: bool = False) -> Tensor:
        ''' Returns indicies tensor denoting minimum values over an axis. '''

        return self._argmin(axis, keepdims)

    def reshape(self, *shape) -> Tensor:
        ''' Reshape/view current tensor with given shape. '''

        return self._reshape(_get_proper_shape(shape))

    def permute(self, *shape):
        ''' Permute dimensions of a tensor. '''

        return self._permute(_get_proper_reduction_axes(shape))

    def transpose(self):
        ''' Transpose the last two dimensions of tensor. '''

        return self._transpose()

    ## METHODS END ##

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

        # Init tensor
        self._op = op
        self._inputs = inputs
        self._data = data
        self._device = device
        self._dtype = dtype
        self._requires_grad = req_grad

        # By default tensor does not retain grad
        self._retain_grad = False

    cdef object _compute_data(self):
        ''' Resolve computational graph computing current tensor data. '''

        if self._data is None:
            self._data = self._op.fwd(
                self,

                # Input x
                (<Tensor> self._inputs.x)._compute_data()
                    if self._inputs.x != NULL else None,

                # Input y
                (<Tensor> self._inputs.y)._compute_data()
                    if self._inputs.y != NULL else None
            )

        return self._data

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

        if Py_TYPE(data) is <PyTypeObject *> Tensor:
            data = (<Tensor> data)._compute_data()
            dtype = (<Tensor> data)._dtype
            shape = (<Tensor> data)._shape
            nshape = (<Tensor> data)._nshape

        res._tensor_init(
            Op(NULL, NULL, OpType.INVALID),
            OpInput(NULL, NULL),
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
        OpInput inputs,
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
        for i in range(ncache):
            res._value_cache[i] = cache[i]

        # Compute the operation if not in lazy state.
        if not _LAZY_STATE:
            res._compute_data()

            # Calling `res._tensor_init()` builds a computational graph by
            # filling in operator structures. If gradient calculation is not
            # required, no need to build a computational graph.
            if not res._requires_grad:
                res._op = Op(NULL, NULL, OpType.INVALID)
                res._inputs = OpInput(NULL, NULL)

        return res

    ## CDEF STATIC METHODS END ##

    ## CDEF OPERATOR METHODS ##

    cdef Tensor _add(self, other: Tensor | any):
        ''' Performs addition with a tensor or a scalar (C ONLY). '''

        # Python scope forward declaration
        cdef OpInput inputs
        cdef Tensor ten
        cdef DType dtype
        cdef _ShapeInfo info

        # Initialize first input
        Py_XINCREF(<PyObject *> self)
        inputs = OpInput(<PyObject *> self, NULL)

        # The `other` object will either be passed as a value cache or as
        # an input.
        Py_XINCREF(<PyObject *> other)

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

        Py_XINCREF(<PyObject *> self)
        cdef OpInput inputs = OpInput(<PyObject *> self, NULL)
        return Tensor._make_from_op(
            _op_negate,
            inputs,
            self._device,
            self._dtype,
            self._shape, self._nshape,
            True,
            NULL, 0
        )

    cdef Tensor _sub(self, other: Tensor | any):
        ''' Subtracts another tensor or a scalar from self tensor (C ONLY). '''

        # Python scope forward declaration
        cdef OpInput inputs
        cdef Tensor ten
        cdef DType dtype
        cdef _ShapeInfo info

        # Initialize first input
        Py_XINCREF(<PyObject *> self)
        inputs = OpInput(<PyObject *> self, NULL)

        # The `other` object will either be passed as a value cache or as
        # an input.
        Py_XINCREF(<PyObject *> other)

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
        cdef OpInput inputs
        cdef Tensor ten
        cdef DType dtype
        cdef _ShapeInfo info

        # Initialize first input
        Py_XINCREF(<PyObject *> self)
        inputs = OpInput(NULL, <PyObject *> self)

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
        Py_XINCREF(<PyObject *> other)
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
        cdef OpInput inputs
        cdef Tensor ten
        cdef DType dtype
        cdef _ShapeInfo info

        # Initialize first input
        Py_XINCREF(<PyObject *> self)
        inputs = OpInput(<PyObject *> self, NULL)

        # The `other` object will either be passed as a value cache or as
        # an input.
        Py_XINCREF(<PyObject *> other)

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
        cdef OpInput inputs
        cdef Tensor ten
        cdef DType dtype
        cdef _ShapeInfo info

        # Initialize first input
        Py_XINCREF(<PyObject *> self)
        inputs = OpInput(<PyObject *> self, NULL)

        # The `other` object will either be passed as a value cache or as
        # an input.
        Py_XINCREF(<PyObject *> other)

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
        cdef OpInput inputs
        cdef Tensor ten
        cdef DType dtype
        cdef _ShapeInfo info

        # Initialize first input
        Py_XINCREF(<PyObject *> self)
        inputs = OpInput(NULL, <PyObject *> self)

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
        Py_XINCREF(<PyObject *> other)
        cdef (PyObject *)[2] value_cache = [
            <PyObject *> 0x00,  # Commute
            <PyObject *> other
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

    cdef Tensor _pow(self, other: Tensor | any):
        ''' Raise a tensor to a power (C ONLY). '''

        # Python scope forward declaration
        cdef OpInput inputs
        cdef Tensor ten
        cdef DType dtype
        cdef _ShapeInfo info

        # Initialize first input
        Py_XINCREF(<PyObject *> self)
        inputs = OpInput(<PyObject *> self, NULL)

        # The `other` object will either be passed as a value cache or as
        # an input.
        Py_XINCREF(<PyObject *> other)

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
        cdef OpInput inputs
        cdef Tensor ten
        cdef DType dtype
        cdef _ShapeInfo info

        # Initialize first input
        Py_XINCREF(<PyObject *> self)
        inputs = OpInput(NULL, <PyObject *> self)

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
        Py_XINCREF(<PyObject *> other)
        cdef (PyObject *)[2] value_cache = [
            <PyObject *> 0x00,  # Commute
            <PyObject *> other
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

        Py_XINCREF(<PyObject *> shape)
        cdef (PyObject *)[1] value_cache = [ <PyObject *> shape ]

        Py_XINCREF(<PyObject *> self)
        return Tensor._make_from_op(
            _op_broadcast_to,
            OpInput(<PyObject *> self, NULL),
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

        Py_XINCREF(<PyObject *> self)
        Py_XINCREF(<PyObject *> axes)
        Py_XINCREF(<PyObject *> keepdims)

        # Value cache
        cdef (PyObject *)[2] value_cache = [
            <PyObject *> axes,
            <PyObject *> keepdims
        ]

        return Tensor._make_from_op(
            _op_sum,
            OpInput(<PyObject *> self, NULL),
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
        cdef int new_shape_idx

        shape, nshape = _validate_and_create_reduction_axes(
            axes,
            self._shape, self._nshape,
            keepdims is True
        )

        # Find the number of observations
        cdef long observations = 1
        if self._nshape == self._nshape:
            for i in range(nshape):
                if self._shape[i] != shape[i]:
                    observations *= self._shape[i]
        else:
            new_shape_idx = 0
            for i in range(self._nshape):
                if self._shape[i] != shape[new_shape_idx]:
                    observations *= self._shape[i]
                else: new_shape_idx += 1

        cdef object py_observations = PyLong_FromLong(observations)

        # Mean over all axis
        if nshape == 0:
            axes = None

        Py_XINCREF(<PyObject *> self)
        Py_XINCREF(<PyObject *> axes)
        Py_XINCREF(<PyObject *> keepdims)
        Py_XINCREF(<PyObject *> py_observations)

        # Value cache
        cdef (PyObject *)[3] value_cache = [
            <PyObject *> axes,
            <PyObject *> keepdims,
            <PyObject *> py_observations
        ]

        return Tensor._make_from_op(
            _op_mean,
            OpInput(<PyObject *> self, NULL),
            self._device,
            dtype,
            shape, nshape,
            False,
            value_cache, 3
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

        Py_XINCREF(<PyObject *> self)
        Py_XINCREF(<PyObject *> axes)
        Py_XINCREF(<PyObject *> keepdims)

        # Value cache
        cdef (PyObject *)[2] value_cache = [
            <PyObject *> axes,
            <PyObject *> keepdims
        ]

        return Tensor._make_from_op(
            _op_max,
            OpInput(<PyObject *> self, NULL),
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

        Py_XINCREF(<PyObject *> self)
        Py_XINCREF(<PyObject *> axes)
        Py_XINCREF(<PyObject *> keepdims)

        # Value cache
        cdef (PyObject *)[2] value_cache = [
            <PyObject *> axes,
            <PyObject *> keepdims
        ]

        return Tensor._make_from_op(
            _op_min,
            OpInput(<PyObject *> self, NULL),
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

        Py_XINCREF(<PyObject *> self)
        Py_XINCREF(<PyObject *> other)
        cdef OpInput inputs = OpInput(<PyObject *> self, <PyObject *> other)

        return Tensor._make_from_op(
            _op_matmul,
            inputs,
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
        for i in range(new_nshape):
            sh = <int> <object> PyTuple_GET_ITEM(shape, i)
            new_shape_size *= sh
            new_shape[i] = sh

        if self_size != new_shape_size:
            free(new_shape)
            raise ValueError(f'Cannot reshape {self_size} elements to {shape}!')

        Py_XINCREF(<PyObject *> self)
        Py_XINCREF(<PyObject *> shape)

        # Caching
        cdef (PyObject *)[1] value_cache = [ <PyObject *> shape ]

        return Tensor._make_from_op(
            _op_reshape,
            OpInput(<PyObject *> self, NULL),
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

        Py_XINCREF(<PyObject *> self)
        Py_XINCREF(<PyObject *> axes)

        # Value cache
        cdef (PyObject *)[1] value_cache = [
            <PyObject *> axes
        ]

        return Tensor._make_from_op(
            _op_permute,
            OpInput(<PyObject *> self, NULL),
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

        Py_XINCREF(<PyObject *> self)

        return Tensor._make_from_op(
            _op_transpose,
            OpInput(<PyObject *> self, NULL),
            self._device,
            self._dtype,
            shape, nshape,
            False,
            NULL, 0
        )

    cdef Tensor _getitem(self, idx: int | slice | tuple[int | slice]):
        ''' Get element(s) from given indicies (C ONLY). '''

        # Python scope forward declarations
        cdef int axis
        cdef object index

        cdef int *shape = <int *> malloc(self._nshape * sizeof(int))
        if shape == NULL:
            raise MemoryError('Failed to create memory for select shape!')
        cdef int nshape = 0

        cdef Py_ssize_t i = 0
        if Py_TYPE(idx) is <PyTypeObject *> int:
            axis = <int> idx

            if axis < -self._shape[i] or axis >= self._shape[i]:
                free(shape)
                raise ValueError(f'Select index out of bounds - {idx}')

            i += 1
        elif Py_TYPE(idx) is <PyTypeObject *> slice:
            shape[nshape] = _get_slice_length(<slice> idx, self._shape[i])
            nshape += 1
            i += 1
        else:  # tuple
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

        # Copy remaining shape
        while i + 1 < self._nshape:  # i + 1: Python for loops are unlike C
            shape[nshape] = self._shape[i]
            nshape += 1
            i += 1

        Py_XINCREF(<PyObject *> self)
        Py_XINCREF(<PyObject *> idx)

        # Value cache
        cdef (PyObject *)[1] value_cache = [
            <PyObject *> idx
        ]

        return Tensor._make_from_op(
            _op_select,
            OpInput(<PyObject *> self, NULL),
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
