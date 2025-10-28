from cpython.object cimport PyObject
from soket.tensor.tensor cimport (Tensor, Op, OpType, TensorTriad,
    _get_proper_reduction_axes, _validate_and_create_reduction_axes)
from soket.tensor.ops cimport (_log_fwd, _log_bwd, _exp_fwd, _exp_bwd,
    _logsumexp_fwd, _logsumexp_bwd)


def log(Tensor x) -> Tensor:
    ''' Perform 2-base logarithm operation on tensor. '''

    return Tensor._make_from_op(
        Op(
            _log_fwd,
            _log_bwd,
            OpType.LOG
        ),
        TensorTriad(<PyObject *> x, NULL, NULL),
        x._device,
        x._dtype,
        x._shape, x._nshape,
        True,
        NULL, 0
    )


def exp(Tensor x) -> Tensor:
    ''' Perform exponential operation on tensor. '''

    return Tensor._make_from_op(
        Op(
            _exp_fwd,
            _exp_bwd,
            OpType.EXP
        ),
        TensorTriad(<PyObject *> x, NULL, NULL),
        x._device,
        x._dtype,
        x._shape, x._nshape,
        True,
        NULL, 0
    )


def logsumexp(Tensor x, *axes: tuple[int], keepdims: bool = False) -> Tensor:
    ''' Performs log-sum-exp operation on the given tensor. '''

    axes = _get_proper_reduction_axes(axes)

    cdef int* shape
    cdef int nshape
    shape, nshape = _validate_and_create_reduction_axes(
        axes,
        x._shape, x._nshape,
        keepdims is True
    )

    # log-sum-exp over all axis
    if nshape == 0:
        axes = None

    # Value cache
    cdef (PyObject *)[2] value_cache = [
        <PyObject *> axes,
        <PyObject *> keepdims
    ]

    return Tensor._make_from_op(
        Op(
            _logsumexp_fwd,
            _logsumexp_bwd,
            OpType.LOGSUMEXP
        ),
        TensorTriad(<PyObject *> x, NULL, NULL),
        x._device,
        x._dtype,
        shape, nshape,
        False,
        value_cache, 2
    )
