## OPERATIONS WHICH CREATES A NEW TENSOR ##

from cpython.object cimport Py_TYPE, PyTypeObject
from soket.tensor cimport (Op, OpType, OpInput, _ShapeInfo,
    _get_shape_info_from_tuple, _get_proper_shape)
from soket.backend cimport _default_device
from soket.dtype cimport _default_datatype
from soket.tensor.ops.intern cimport _max


## HELPER FUNCTIONS ##

cdef inline Tensor _create_and_return_tensor(
    tuple shape,
    object data,
    Device device,
    DType dtype,
    requires_grad
):
    # Deduce shape from tuple
    cdef _ShapeInfo info = _get_shape_info_from_tuple(shape)

    cdef Tensor res = Tensor.__new__(Tensor)
    res._tensor_init(
        Op(NULL, NULL, OpType.INVALID),
        OpInput(NULL, NULL),
        data,
        device,
        dtype,
        info.shape, info.nshape,
        False,
        requires_grad
    )

    return res

## HELPER FUNCTIONS END ##


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
