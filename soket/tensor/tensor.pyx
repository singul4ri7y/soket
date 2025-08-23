from cpython.object cimport Py_TYPE, PyTypeObject
from cpython.tuple cimport PyTuple_GET_SIZE, PyTuple_GET_ITEM
from cpython.long cimport PyLong_AsLong, PyLong_AsSsize_t, PyLong_FromSsize_t
from cython cimport freelist
from soket.backend cimport _default_device, _is_gpu_available, DeviceType
from soket.dtype cimport _default_datatype, _get_scalar_dtype, promote_types
from libc.string cimport memcpy
from libc.stdlib cimport malloc, free
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


## HELPER DATA-STRUCTURES AND FUNCTIONS ##


## OPS ##

cdef Op _elemwise_add = Op(
    _elemwise_add_fwd,
    NULL,
    OpType.ELEMWISE_ADD
)

## OPS END ##


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


cdef struct _ShapeInfo:
    # Contains shape information.
    # The `shape` pointer is heap allocated, use with caution.

    int *shape
    int nshape


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
        raise RuntimeError(f'Could not allocate memory to store shape!')

    for i in range(info.nshape):
        info.shape[i] = PyLong_AsLong(<object> PyTuple_GET_ITEM(shape, i))
    
    return info
    

cdef inline _ShapeInfo _get_broadcasted_shape(
    int *x_shape,
    int x_nshape,

    int *y_shape,
    int y_nshape
) noexcept:
    ''' Get resulting broadcasted shape from two shapes. '''

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
        raise RuntimeError(f'Could not allocate memory for shape!')

    info.nshape = max_nshape

    # Smaller length shapes filled with 1 at the beginning
    cdef Py_ssize_t i
    for i in range(diff):
        info.shape[i] = max_shape[i]
    
    for i in range(min_nshape):
        mis = min_shape[i]
        mas = max_shape[i + diff]

        info.shape[i + diff] = mas if mas > mis else mis
    
    return info


cdef void _check_device_compatibility(Device a, Device b):
    '''
    Check for device compatibility.

    Raises error if devices are not compatible.
    '''
    
    if a._ne(b):
        raise RuntimeError(f'Incompatible devices - {a} and {b}')

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
    
    ## DUNDER METHODS ##

    def __str__(self) -> str:
        ''' String representation of a tensor. '''

        return self._data.__str__()
    

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

    ## PROPERTIES END ##
    
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
                raise RuntimeError(f'Could not allocate memory to store shape!')
            
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

                (<Tensor> self._inputs.x)._compute_data() 
                    if self._inputs.x != NULL else None,
                (<Tensor> self._inputs.x)._shape 
                    if self._inputs.x != NULL else NULL,
                (<Tensor> self._inputs.x)._nshape 
                    if self._inputs.x != NULL else -1,

                (<Tensor> self._inputs.y)._compute_data() 
                    if self._inputs.y != NULL else None,
                (<Tensor> self._inputs.y)._shape 
                    if self._inputs.y != NULL else NULL,
                (<Tensor> self._inputs.y)._nshape 
                    if self._inputs.y != NULL else -1,
            )

        return self._data

    ## CDEF METHODS END ##

    ## CDEF STATIC METHODS ##

    @staticmethod
    cdef Tensor _make_const(
        data,
        Device device,
        DType dtype,
        int *shape, int nshape,
        requires_grad
    ):
        '''
        Generic construction function for creating detached tensors.

        This routine does not check for valid device, use with caution.
        '''

        # Forward delcaration for Python scope
        cdef _ShapeInfo info

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
            True,
            requires_grad
        )

        return res
    
    @staticmethod
    cdef Tensor _make_from_op(
        Op op,
        OpInput inputs,
        DType dtype,
        int *shape, int nshape
    ):
        ''' Constructs a new tensor from an operation. '''

        # Create Tensor object w/o calling the constructor
        cdef Tensor res = Tensor.__new__(Tensor)
        res._tensor_init(
            op,
            inputs,
            None,
            (<Tensor> inputs.x)._device,
            dtype,
            shape, nshape,
            False,
            None
        )

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
        ''' Performs addition with a tensor or a scalar. '''

        cdef OpInput inputs = OpInput(<PyObject *> self, NULL)
        cdef DType dtype
        cdef _ShapeInfo info

        if Py_TYPE(other) is <PyTypeObject *> Tensor:
            # Check if devices are compatible.
            _check_device_compatibility(self._device, (<Tensor> other)._device)

            inputs.y = <PyObject *> other

            if self._dtype._eq((<Tensor> other)._dtype):
                dtype = self._dtype
            else:
                dtype = promote_types(self._dtype, (<Tensor> other)._dtype)
            info = _get_broadcasted_shape(
                self._shape,
                self._nshape,

                (<Tensor> other)._shape,
                (<Tensor> other)._nshape
            )

            return Tensor._make_from_op(
                _elemwise_add,
                inputs,
                dtype,
                info.shape,
                info.nshape
            )

    ## CDEF OPERATOR METHODS END ##

    ## DUNDER OPERATORS ##

    def __add__(self, other: Tensor | any) -> Tensor:
        ''' Performs addition with another tensor/scalar. '''

        return self._add(other)

    ## DUNDER OPERATORS END ##