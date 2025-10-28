## FORWARD PASS OPERATIONS ##

from cpython.ref cimport Py_XINCREF
from soket.tensor.ops.intern cimport *


cdef object _elemwise_add_fwd(Tensor target, object x, object y, object Z):
    ''' Performs element-wise addition on the data. '''

    return _add(target._device._dev_idx)(
        x, y,
        dtype=target._dtype._str()
    )


cdef object _scalar_add_fwd(Tensor target, object x, object y, object Z):
    ''' Performs scalar addition on the data. '''

    return _add(target._device._dev_idx)(
        x, <object> target._value_cache[0],  # Scalar
        dtype=target._dtype._str()
    )


cdef object _negate_fwd(Tensor target, object x, object y, object Z):
    ''' Performs negation on the data. '''

    return _neg(target._device._dev_idx)(x)


cdef object _elemwise_sub_fwd(Tensor target, object x, object y, object Z):
    ''' Performs element-wise subtraction on the data. '''

    return _sub(target._device._dev_idx)(
        x, y,
        dtype=target._dtype._str()
    )


cdef object _scalar_sub_fwd(Tensor target, object x, object y, object Z):
    ''' Performs scalar subtraction on the data. '''

    if <int> target._value_cache[1] == 0:  # Don't commute
        return _sub(target._device._dev_idx)(
            x, <object> target._value_cache[0],  # Scalar
            dtype=target._dtype._str()
        )

    return _sub(target._device._dev_idx)(
        <object> target._value_cache[1], y,
        dtype=target._dtype._str()
    )

cdef object _elemwise_mul_fwd(Tensor target, object x, object y, object Z):
    ''' Performs element-wise multiplication on the data. '''

    return _mul(target._device._dev_idx)(
        x, y,
        dtype=target._dtype._str()
    )


cdef object _scalar_mul_fwd(Tensor target, object x, object y, object Z):
    ''' Performs scalar multiplication on the data. '''

    return _mul(target._device._dev_idx)(
        x, <object> target._value_cache[0],  # Scalar
        dtype=target._dtype._str()
    )


cdef object _elemwise_div_fwd(Tensor target, object x, object y, object Z):
    ''' Performs element-wise division on the data. '''

    return _div(target._device._dev_idx)(
        x, y,
        dtype=target._dtype._str()
    )


cdef object _scalar_div_fwd(Tensor target, object x, object y, object Z):
    ''' Performs scalar division on the data. '''

    if <int> target._value_cache[1] == 0:  # Don't commute
        return _div(target._device._dev_idx)(
            x, <object> target._value_cache[0],  # Scalar
            dtype=target._dtype._str()
        )

    return _div(target._device._dev_idx)(
        <object> target._value_cache[1], y,
        dtype=target._dtype._str()
    )


cdef object _elemwise_pow_fwd(Tensor target, object x, object y, object Z):
    ''' Raises data to a power. '''

    return _pow(target._device._dev_idx)(
        x, y,
        dtype=target._dtype._str()
    )


cdef object _scalar_pow_fwd(Tensor target, object x, object y, object Z):
    ''' Raises data to a scalar or vice versa. '''

    if <int> target._value_cache[1] == 0:  # Don't commute
        return _pow(target._device._dev_idx)(
            x, <object> target._value_cache[0],  # Scalar
            dtype=target._dtype._str()
        )

    return _pow(target._device._dev_idx)(
        <object> target._value_cache[1], y,
        dtype=target._dtype._str()
    )


cdef object _broadcast_to_fwd(Tensor target, object x, object y, object Z):
    ''' Broadcasts data to given shape. '''

    return _broadcast_to(target._device._dev_idx)(
        x, <object> target._value_cache[0]
    )


cdef object _sum_fwd(Tensor target, object x, object y, object Z):
    ''' Performs summation reduction over the data. '''

    return _sum(target._device._dev_idx)(
        x,
        <object> target._value_cache[0],
        target._dtype._str(),
        None,
        <object> target._value_cache[1]
    )


cdef object _mean_fwd(Tensor target, object x, object y, object Z):
    ''' Performs mean reduction over the data. '''

    return _mean(target._device._dev_idx)(
        x,
        <object> target._value_cache[0],
        target._dtype._str(),
        None,
        <object> target._value_cache[1]
    )


cdef object _max_fwd(Tensor target, object x, object y, object Z):
    ''' Find maximum value(s) over given axes of data. '''

    return _max(target._device._dev_idx)(
        x,
        <object> target._value_cache[0],
        None,
        <object> target._value_cache[1]
    )

cdef object _min_fwd(Tensor target, object x, object y, object Z):
    ''' Find minimum value(s) over given axes of data. '''

    return _min(target._device._dev_idx)(
        x,
        <object> target._value_cache[0],
        None,
        <object> target._value_cache[1]
    )

cdef object _matmul_fwd(Tensor target, object x, object y, object Z):
    ''' Performs matrix-matrix multiply operation on data. '''

    return _matmul(target._device._dev_idx)(
        x, y,
        dtype=target._dtype._str()
    )

cdef object _reshape_fwd(Tensor target, object x, object y, object Z):
    ''' Performs reshape operation on data. '''

    return _reshape(target._device._dev_idx)(
        x,
        <object> target._value_cache[0]
    )

cdef object _permute_fwd(Tensor target, object x, object y, object Z):
    ''' Permute the dimensions of the data. '''

    return _transpose(target._device._dev_idx)(
        x,
        <object> target._value_cache[0]
    )

cdef object _transpose_fwd(Tensor target, object x, object y, object Z):
    ''' Transpose a data (last two dimensions swap). '''

    return x.T

cdef object _select_fwd(Tensor target, object x, object y, object Z):
    ''' Select values from data. '''

    return x.__getitem__(<object> target._value_cache[0])

cdef object _relu_fwd(Tensor target, object x, object y, object Z):
    ''' Perform ReLU activation operation on data. '''

    return _maximum(target._device._dev_idx)(x, 0)


cdef object _log_fwd(Tensor target, object x, object y, object Z):
    ''' Perform 2-base log on data. '''

    return _log(target._device._dev_idx)(x)


cdef object _exp_fwd(Tensor target, object x, object y, object Z):
    ''' Perform exponential operation on the data. '''

    return _exp(target._device._dev_idx)(x)


cdef object _logsumexp_fwd(Tensor target, object x, object y, object Z):
    ''' Perform log-sum-exp operation on the data. '''

    cdef int didx = target._device._dev_idx
    cdef object axes = <object> target._value_cache[0]

    # Maximum value for numerical stability
    cdef object max = _max(didx)(x, axes, None, True)
    cdef object res = _add(didx)(
        _log(didx)(_sum(didx)(
            _exp(didx)(
                _sub(didx)(x, max)
            ),
            axes,
            None, None,
            True
        )),
        max
    )

    if <object> target._value_cache[1] is not True:
        res = _squeeze(didx)(res, axes)

    return res


cdef object _sxentropyloss_fwd(Tensor target, object x, object y, object Z):
    ''' Perform softmax cross entropy operation on the data. '''

    cdef object one_hot = <object> target._value_cache[2]
    cdef int didx = target._device._dev_idx

    # Find cross entropy loss over the whole batch
    cdef object batch_xentropy = _sub(didx)(
        _logsumexp_fwd(target, x, y, None),
        _sum(didx)(
            _mul(didx)(x, one_hot, dtype=target._dtype._str()),
            <object> target._value_cache[0]  # reduction axes
        )
    )

    cdef str reduction = <str> target._value_cache[3]
    if reduction == 'sum':
        return _sum(didx)(batch_xentropy, (0,))
    elif reduction == 'mean':
        return _mean(didx)(batch_xentropy, (0,))

    return batch_xentropy


cdef object _bnorm_fwd(Tensor target, object x, object y, object Z):
    ''' Forward pass of batch normalization. '''

    cdef object reduce_axes = <object> target._value_cache[0]
    # No. of observations -> target._value_cache[1]
    cdef Tensor running_mean = <Tensor> target._value_cache[2]
    cdef Tensor running_var = <Tensor> target._value_cache[3]
    cdef bint training = <bint> target._value_cache[4]
    cdef object momentum = <object> target._value_cache[5]
    cdef object eps = <object> target._value_cache[6]
    cdef int didx = target._device._dev_idx
    cdef bint track_running_stats = running_mean is not None
    cdef bint layernorm = not track_running_stats and momentum is None
    cdef sub_momentum

    # For intermediate computations
    cdef object xshift, var
    cdef object mean, rvar, norm  # Might be cached

    if training is True or (training is False and track_running_stats is False):
        # mean
        mean = _mean(didx)(Z, reduce_axes, None, None, True)
        xshift = _sub(didx)(Z, mean)
        # variance
        var = _mean(didx)(
            _pow(didx)(xshift, 2),
            reduce_axes, None, None, True
        )
    else:
        mean = running_mean
        var = running_var
        xshift = _sub(didx)(Z, mean)

    # Update running mean and variance
    if training is True and track_running_stats is True:
        sub_momentum = 1.0 - momentum

        running_mean._data_ = _add(didx)(
            _mul(didx)(running_mean._compute_data(), sub_momentum),
            _mul(didx)(mean, momentum)
        )
        running_var._data_ = _add(didx)(
            _mul(didx)(running_var._compute_data(), sub_momentum),
            _mul(didx)(var, momentum)
        )

    # Reciprocal of variance
    rvar = _pow(didx)(
        _add(didx)(var, eps),
        -0.5
    )
    norm = _mul(didx)(xshift, rvar)

    # Cache data for backward pass
    Py_XINCREF(<PyObject *> xshift)
    Py_XINCREF(<PyObject *> rvar)
    Py_XINCREF(<PyObject *> norm)

    target._value_cache[7] = <PyObject *> xshift
    target._value_cache[8] = <PyObject *> rvar
    target._value_cache[9] = <PyObject *> norm

    # x -> gamma, y -> beta
    if x is None:
        return norm

    cdef tuple reduce_shape = <tuple> mean.shape
    # LayerNorm tweak
    if layernorm:
        norm = _mul(didx)(x, norm)
    else:
        norm = _mul(didx)(_reshape(didx)(x, reduce_shape), norm)

    if y is None:
        return norm

    # LayerNorm tweak
    if layernorm:
        return _add(didx)(y, norm)
    return _add(didx)(_reshape(didx)(y, reduce_shape), norm)
