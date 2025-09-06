## FORWARD PASS OPERATIONS ##

from soket.tensor.ops.intern cimport *


cdef object _elemwise_add_fwd(Tensor target, object x, object y):
    ''' Performs element-wise addition on the data. '''

    return _add(target._device._dev_idx)(
        x, y,
        dtype=target._dtype._str() if target._dtype is not None else None
    )


cdef object _scalar_add_fwd(Tensor target, object x, object y):
    ''' Performs scalar addition on the data. '''

    return _add(target._device._dev_idx)(
        x, <object> target._value_cache[0],  # Scalar
        dtype=target._dtype._str() if target._dtype is not None else None
    )


cdef object _negate_fwd(Tensor target, object x, object y):
    ''' Performs negation on the data. '''

    return _neg(target._device._dev_idx)(x)


cdef object _elemwise_sub_fwd(Tensor target, object x, object y):
    ''' Performs element-wise subtraction on the data. '''

    return _sub(target._device._dev_idx)(
        x, y,
        dtype=target._dtype._str() if target._dtype is not None else None
    )


cdef object _scalar_sub_fwd(Tensor target, object x, object y):
    ''' Performs scalar subtraction on the data. '''

    if <int> target._value_cache[1] == 0:  # Don't commute
        return _sub(target._device._dev_idx)(
            x, <object> target._value_cache[0],  # Scalar
            dtype=target._dtype._str() if target._dtype is not None else None
        )

    return _sub(target._device._dev_idx)(
        <object> target._value_cache[1], y,
        dtype=target._dtype._str() if target._dtype is not None else None
    )

cdef object _elemwise_mul_fwd(Tensor target, object x, object y):
    ''' Performs element-wise multiplication on the data. '''

    return _mul(target._device._dev_idx)(
        x, y,
        dtype=target._dtype._str() if target._dtype is not None else None
    )


cdef object _scalar_mul_fwd(Tensor target, object x, object y):
    ''' Performs scalar multiplication on the data. '''

    return _mul(target._device._dev_idx)(
        x, <object> target._value_cache[0],  # Scalar
        dtype=target._dtype._str() if target._dtype is not None else None
    )


cdef object _elemwise_div_fwd(Tensor target, object x, object y):
    ''' Performs element-wise division on the data. '''

    return _div(target._device._dev_idx)(
        x, y,
        dtype=target._dtype._str() if target._dtype is not None else None
    )


cdef object _scalar_div_fwd(Tensor target, object x, object y):
    ''' Performs scalar division on the data. '''

    if <int> target._value_cache[1] == 0:  # Don't commute
        return _div(target._device._dev_idx)(
            x, <object> target._value_cache[0],  # Scalar
            dtype=target._dtype._str() if target._dtype is not None else None
        )

    return _div(target._device._dev_idx)(
        <object> target._value_cache[1], y,
        dtype=target._dtype._str() if target._dtype is not None else None
    )


cdef object _elemwise_pow_fwd(Tensor target, object x, object y):
    ''' Raises data to a power. '''

    return _pow(target._device._dev_idx)(
        x, y,
        dtype=target._dtype._str() if target._dtype is not None else None
    )


cdef object _scalar_pow_fwd(Tensor target, object x, object y):
    ''' Raises data to a scalar or vice versa. '''

    if <int> target._value_cache[1] == 0:  # Don't commute
        return _pow(target._device._dev_idx)(
            x, <object> target._value_cache[0],  # Scalar
            dtype=target._dtype._str() if target._dtype is not None else None
        )

    return _pow(target._device._dev_idx)(
        <object> target._value_cache[1], y,
        dtype=target._dtype._str() if target._dtype is not None else None
    )


cdef object _broadcast_to_fwd(Tensor target, object x, object y):
    ''' Broadcasts data to given shape. '''

    return _broadcast_to(target._device._dev_idx)(
        x, <object> target._value_cache[0]
    )


cdef object _sum_fwd(Tensor target, object x, object y):
    ''' Performs summation reduction over the data. '''

    return _sum(target._device._dev_idx)(
        x,
        <object> target._value_cache[0],
        target._dtype._str(),
        None,
        <object> target._value_cache[1]
    )


cdef object _mean_fwd(Tensor target, object x, object y):
    ''' Performs mean reduction over the data. '''

    return _mean(target._device._dev_idx)(
        x,
        <object> target._value_cache[0],
        target._dtype._str(),
        None,
        <object> target._value_cache[1]
    )


cdef object _max_fwd(Tensor target, object x, object y):
    ''' Find maximum value(s) over given axes of data. '''

    return _max(target._device._dev_idx)(
        x,
        <object> target._value_cache[0],
        None,
        <object> target._value_cache[1]
    )

cdef object _min_fwd(Tensor target, object x, object y):
    ''' Find minimum value(s) over given axes of data. '''

    return _min(target._device._dev_idx)(
        x,
        <object> target._value_cache[0],
        None,
        <object> target._value_cache[1]
    )

cdef object _matmul_fwd(Tensor target, object x, object y):
    ''' Performs matrix-matrix multiply operation on data. '''

    return _matmul(target._device._dev_idx)(
        x, y,
        dtype=target._dtype._str()
    )

cdef object _reshape_fwd(Tensor target, object x, object y):
    ''' Performs reshape operation on data. '''

    return _reshape(target._device._dev_idx)(
        x,
        <object> target._value_cache[0]
    )

cdef object _permute_fwd(Tensor target, object x, object y):
    ''' Permute the dimensions of the data. '''

    return _transpose(target._device._dev_idx)(
        x,
        <object> target._value_cache[0]
    )

cdef object _transpose_fwd(Tensor target, object x, object y):
    ''' Transpose a data (last two dimensions swap). '''

    return x.T

cdef object _select_fwd(Tensor target, object x, object y):
    ''' Select values from data. '''

    return x.__getitem__(<object> target._value_cache[0])
