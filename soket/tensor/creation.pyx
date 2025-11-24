## OPERATIONS WHICH CREATES A NEW TENSOR ##

from cpython.object cimport Py_TYPE, PyTypeObject
from soket.tensor cimport (Op, OpType, TensorTriad, _ShapeInfo,
    _get_shape_info_from_tuple, _get_proper_shape)
from soket.backend cimport _default_device
from soket.dtype cimport _default_datatype, int32
from soket.tensor.ops.intern cimport *
from libc.stdlib cimport malloc


## HELPER FUNCTIONS ##

cdef inline Tensor _create_and_return_tensor(
    tuple shape,
    object data,
    Device device,
    DType dtype,
    requires_grad
):
    # Deduce shape from tuple
    cdef _ShapeInfo info
    if shape == None:  # scalar tensor
        info.shape = <int *> malloc(0)
        info.nshape = 0
    else:
        info = _get_shape_info_from_tuple(shape)

    return Tensor._make_const(
        data,
        device,
        dtype,
        info.shape, info.nshape,
        False,
        requires_grad
    )

## HELPER FUNCTIONS END ##


## CDEF METHODS ##

cdef Tensor _randb(tuple shape, p, Device device, DType dtype, requires_grad):
    '''
    Returns tensor of given shape filled with samples from a binomial
    distribution of given probability and single trial per sample.

    (C ONLY)
    '''

    return _create_and_return_tensor(
        shape,
        device._randb(shape, p, dtype._str()),
        device,
        dtype,
        requires_grad
    )


cdef Tensor _zeros(tuple shape, Device device, DType dtype, requires_grad):
    ''' Returns tensor filled with zeros of given shape (C ONLY). '''

    return _create_and_return_tensor(
        shape,
        device._zeros(shape, dtype._str()),
        device,
        dtype,
        requires_grad
    )


cdef Tensor _zeros_like(
    Tensor ten,
    Device device,
    DType dtype,
    bint requires_grad
):
    '''
    Returns a tensor of the same shape as given tensor but filled with zeros.

    C ONLY.
    '''

    cdef tuple shape = <tuple> ten.shape
    return _create_and_return_tensor(
        shape,
        device._zeros(shape, dtype._str()),
        device,
        dtype,
        requires_grad
    )


cdef Tensor _ones_like(
    Tensor ten,
    Device device,
    DType dtype,
    bint requires_grad
):
    '''
    Returns a tensor of the same shape as given tensor but filled with ones.

    C ONLY.
    '''

    cdef tuple shape = <tuple> ten.shape
    return _create_and_return_tensor(
        shape,
        device._ones(shape, dtype._str()),
        device,
        dtype,
        requires_grad
    )


cdef Tensor _ones(tuple shape, Device device, DType dtype, requires_grad):
    ''' Returns tensor filled with ones of given shape (C ONLY). '''

    return _create_and_return_tensor(
        shape,
        device._ones(shape, dtype._str()),
        device,
        dtype,
        requires_grad
    )


cdef Tensor _one_hot(
    Tensor tensor,
    int num_classes,
    Device device,
    DType dtype,
    requires_grad
):
    ''' Returns one-hot encoded tensor. '''

    # Refer to `dtype.pyx:_supported_dtypes` for datatype indexing reference.
    if tensor._dtype._idx < 3 or tensor._dtype._idx > 10:
        raise TypeError('Tensor indicies must be integer!')

    tensor = tensor.to(device)
    array = device._one_hot(tensor._compute_data(), num_classes, dtype._str())

    return _create_and_return_tensor(
        <tuple> array.shape,
        array,
        device,
        dtype,
        requires_grad
    )

## CDEF METHODS END ##


def rand(
    *shape,
    object low=0.0,
    object high=1.0,
    Device device=None,
    DType dtype=None,
    requires_grad: bool = None
) -> Tensor:
    '''
    Returns tensor of given shape filled with samples from uniform
    distribution of interval [low, high).
    '''

    shape = _get_proper_shape(shape)

    if (Py_TYPE(low) is not <PyTypeObject *> float or
    Py_TYPE(high) is not <PyTypeObject *> float):
        raise TypeError(f'Invalid low and high types!')

    # Use defaults
    dtype = _default_datatype if dtype is None else dtype
    device = _default_device() if device is None else device

    return _create_and_return_tensor(
        shape,
        device._rand(shape, low, high, dtype._str()),
        device,
        dtype,
        requires_grad
    )


def randn(
    *shape,
    object mean=0.0,
    object std=1.0,
    Device device=None,
    DType dtype=None,
    requires_grad: bool = None
) -> Tensor:
    '''
    Returns tensor of given shape filled with samples from a normal
    distribution of provided mean and variance.
    '''

    shape = _get_proper_shape(shape)

    if (Py_TYPE(mean) is not <PyTypeObject *> float or
    Py_TYPE(std) is not <PyTypeObject *> float):
        raise TypeError('Invalid mean and std types!')

    # Use defaults
    dtype = _default_datatype if dtype is None else dtype
    device = _default_device() if device is None else device

    return _create_and_return_tensor(
        shape,
        device._randn(shape, mean, std, dtype._str()),
        device,
        dtype,
        requires_grad
    )


def randb(
    *shape,
    object p=0.5,
    Device device=None,
    DType dtype=None,
    requires_grad: bool = None
) -> Tensor:
    '''
    Returns tensor of given shape filled with samples from a binomial
    distribution of given probability and single trial per sample.
    '''

    shape = _get_proper_shape(shape)

    if Py_TYPE(p) is not <PyTypeObject *> float:
        raise TypeError('Invalid `p` type! Expected float.')

    # Use defaults
    dtype = _default_datatype if dtype is None else dtype
    device = _default_device() if device is None else device

    return _create_and_return_tensor(
        shape,
        device._randb(shape, p, dtype._str()),
        device,
        dtype,
        requires_grad
    )


def randint(
    object low,
    object high,
    tuple size=None,
    Device device=None,
    DType dtype=None,
    requires_grad: bool = None
) -> Tensor:
    '''
    Returns tensor of given shape filled with samples from a binomial
    distribution of given probability and single trial per sample.
    '''

    # Use defaults
    dtype = int32 if dtype is None else dtype
    device = _default_device() if device is None else device

    return _create_and_return_tensor(
        size,
        device._randint(low, high, size, dtype._str()),
        device,
        dtype,
        requires_grad
    )


def zeros(
    *shape,
    Device device=None,
    DType dtype=None,
    requires_grad: bool = None
) -> Tensor:
    ''' Returns tensor filled with zeros of given shape. '''

    shape = _get_proper_shape(shape)

    # Use defaults
    dtype = _default_datatype if dtype is None else dtype
    device = _default_device() if device is None else device

    return _create_and_return_tensor(
        shape,
        device._zeros(shape, dtype._str()),
        device,
        dtype,
        requires_grad
    )




def zeros_like(
    Tensor ten,
    Device device=None,
    DType dtype=None,
    requires_grad=None
) -> Tensor:
    '''
    Returns a tensor of the same shape as given tensor but filled with zeros.
    '''

    # Use defaults
    dtype = ten._dtype if dtype is None else dtype
    device = ten._device if device is None else device

    cdef tuple shape = <tuple> ten.shape
    return _create_and_return_tensor(
        shape,
        device._zeros(shape, dtype._str()),
        device,
        dtype,
        requires_grad
    )




def ones(
    *shape,
    Device device=None,
    DType dtype=None,
    requires_grad: bool = None
) -> Tensor:
    ''' Returns tensor filled with ones. '''

    shape = _get_proper_shape(shape)

    # Use defaults
    dtype = _default_datatype if dtype is None else dtype
    device = _default_device() if device is None else device

    return _create_and_return_tensor(
        shape,
        device._ones(shape, dtype._str()),
        device,
        dtype,
        requires_grad
    )




def one_like(
    Tensor ten,
    Device device=None,
    DType dtype=None,
    requires_grad=None
) -> Tensor:
    '''
    Returns a tensor of the same shape as given tensor but filled with ones.
    '''

    # Use defaults
    dtype = ten._dtype if dtype is None else dtype
    device = ten._device if device is None else device

    cdef tuple shape = <tuple> ten.shape
    return _create_and_return_tensor(
        shape,
        device._ones(shape, dtype._str()),
        device,
        dtype,
        requires_grad
    )


def empty(
    *shape,
    Device device=None,
    DType dtype=None,
    requires_grad: bool = None
) -> Tensor:
    ''' Returns an uninitialized empty tensor of given shape. '''

    shape = _get_proper_shape(shape)

    # Use defaults
    dtype = _default_datatype if dtype is None else dtype
    device = _default_device() if device is None else device

    return _create_and_return_tensor(
        shape,
        device._empty(shape, dtype._str()),
        device,
        dtype,
        requires_grad
    )


def full(
    *shape,
    object fill=1.0,
    Device device=None,
    DType dtype=None,
    requires_grad: bool = None
) -> Tensor:
    ''' Returns tensor of given shape filled with given value. '''

    shape = _get_proper_shape(shape)

    if (Py_TYPE(fill) is not <PyTypeObject *> int and
    Py_TYPE(fill) is not <PyTypeObject *> float and
    Py_TYPE(fill) is not <PyTypeObject *> bool):
        raise TypeError('Invalid fill type!')

    # Use defaults
    dtype = _default_datatype if dtype is None else dtype
    device = _default_device() if device is None else device

    return _create_and_return_tensor(
        shape,
        device._full(shape, fill, dtype._str()),
        device,
        dtype,
        requires_grad
    )


def linspace(
    object start, object end,
    object num=100,
    object endpoint=True,
    object axis=0,
    Device device=None,
    DType dtype=None,
    requires_grad=None
):
    '''
    Returns a tensor filled with evenly spaced numbers over a specific
    interval.
    '''

    # Extract data if tensor is provided
    if Py_TYPE(start) is <PyTypeObject *> Tensor:
        start = (<Tensor> start)._compute_data()

        if Py_TYPE(end) is not <PyTypeObject *> Tensor:
            raise ValueError('Both start and end point has to be tensors!')
        end = (<Tensor> end)._compute_data()

    dtype = _default_datatype if dtype is None else dtype
    device = _default_device() if device is None else device

    cdef object data = _linspace(device._dev_idx)(
        start, end, num, endpoint,
        False, dtype._str(), axis
    )
    return _create_and_return_tensor(
        data.shape,
        data,
        device,
        dtype,
        requires_grad
    )

# TODO: Move to soket.nn.functional
def one_hot(
    Tensor tensor,
    object num_classes = -1,
    Device device=None,
    DType dtype=None,
    requires_grad=None
) -> Tensor:
    ''' Returns one-hot encoded tensor. '''

    # Refer to `dtype.pyx:_supported_dtypes` for datatype indexing reference.
    if tensor._dtype._idx < 3 or tensor._dtype._idx > 10:
        raise TypeError('Tensor indicies must be integer!')

    dtype = _default_datatype if dtype is None else dtype
    device = _default_device() if device is None else device
    tensor = tensor.to(device)

    cdef object array = tensor._compute_data()
    if num_classes == -1:
        num_classes = _max(device._dev_idx)(array).item() + 1

    array = device._one_hot(array, num_classes, dtype._str())
    return _create_and_return_tensor(
        <tuple> array.shape,
        array,
        device,
        dtype,
        requires_grad
    )
