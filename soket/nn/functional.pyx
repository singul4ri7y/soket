from cpython.object cimport PyObject
from cpython.long cimport PyLong_FromLong
from cpython.tuple cimport PyTuple_New, PyTuple_GET_ITEM, PyTuple_SET_ITEM
from soket.tensor cimport Op, OpType, TensorTriad
from soket.tensor.ops cimport _bnorm_fwd, _bnorm_bwd


cpdef Tensor batch_norm(
    Tensor X,
    Tensor running_mean,
    Tensor running_var,
    Tensor gamma=None,
    Tensor beta=None,
    object training=False,
    object momentum=0.1,
    object eps=1e-5
):
    ''' A generalized Batch Normalization function. '''
    ''' Original paper: https://arxiv.org/abs/1502.03167 '''

    # Check for valid input dimension
    if X._nshape < 2:
        raise ValueError('Expected input tensor dimension to be atleast 2D!')

    # Check for device compatibility
    cdef bint trs = running_mean is not None
    if ((trs and X._device._ne(running_mean._device)) or
        (trs and X._device._ne(running_var._device)) or
        (gamma is not None and X._device._ne(gamma._device)) or
        (beta is not None and X._device._ne(beta._device))
    ):
        raise RuntimeError('Incompatible input and parameter devices!')

    # Check for datatype
    cdef int didx = X._dtype._idx
    if (X._dtype._idx != didx or
        (trs and running_mean._dtype._idx != didx) or
        (trs and running_var._dtype._idx != didx) or
        (gamma is not None and gamma._dtype._idx != didx) or
        (beta is not None and beta._dtype._idx != didx)
    ):
        raise RuntimeError('Input and parameters should share same datatype!')

    cdef tuple reduce_axes = PyTuple_New(X._nshape - 1)
    PyTuple_SET_ITEM(reduce_axes, 0, 0)

    cdef int observations = X._shape[0]
    cdef Py_ssize_t i
    for i in range(1, X._nshape - 1):
        PyTuple_SET_ITEM(reduce_axes, i, i + 1)
        observations *= X._shape[i + 1]

    cdef object py_observations = PyLong_FromLong(observations)

    # Prepare cache. More size is needed due to more forward pass caching.
    cdef (PyObject *)[10] value_cache = [
        <PyObject *> reduce_axes,
        <PyObject *> py_observations,
        <PyObject *> running_mean,
        <PyObject *> running_var,
        <PyObject *> training,
        <PyObject *> momentum,
        <PyObject *> eps,
        NULL, NULL, NULL
    ]

    return Tensor._make_from_op(
        Op(
            _bnorm_fwd,
            _bnorm_bwd,
            OpType.BATCHNORM
        ),
        TensorTriad(<PyObject *> gamma, <PyObject *> beta, <PyObject *> X),
        X._device,
        X._dtype,
        X._shape, X._nshape,
        True,
        value_cache, 10
    )


cpdef Tensor layer_norm(
    Tensor X,
    Tensor weight=None,
    Tensor bias=None,
    object eps=1e-5
):
    ''' A generalized Layer Normalization function. '''
    ''' Original paper: https://arxiv.org/abs/1608.06450 '''

    # Check for valid input dimension
    if X._nshape < 2:
        raise ValueError('Expected input tensor dimension to be atleast 2D!')

    # Check for device compatibility
    if ((weight is not None and X._device._ne(weight._device)) or
        (bias is not None and X._device._ne(bias._device))
    ):
        raise RuntimeError('Incompatible input and parameter devices!')

    # Check for datatype
    cdef int didx = X._dtype._idx
    if (X._dtype._idx != didx or
       (weight is not None and weight._dtype._idx != didx) or
       (bias is not None and bias._dtype._idx != didx)
    ):
        raise RuntimeError('Input and parameters should share same datatype!')

    cdef tuple reduce_axes = PyTuple_New(X._nshape - 1)

    cdef int observations = 1
    cdef Py_ssize_t i
    for i in range(1, X._nshape):
        PyTuple_SET_ITEM(reduce_axes, i - 1, i)
        observations *= X._shape[i]

    cdef object py_observations = PyLong_FromLong(observations)

    cdef object training = True
    # Prepare cache. More size is needed due to more forward pass caching.
    cdef (PyObject *)[10] value_cache = [
        <PyObject *> reduce_axes,
        <PyObject *> py_observations,
        <PyObject *> None,
        <PyObject *> None,
        <PyObject *> training,  # training = True
        <PyObject *> None,  # don't care
        <PyObject *> eps,
        NULL, NULL, NULL
    ]

    return Tensor._make_from_op(
        Op(
            _bnorm_fwd,
            _bnorm_bwd,
            OpType.BATCHNORM
        ),
        TensorTriad(<PyObject *> weight, <PyObject *> bias, <PyObject *> X),
        X._device,
        X._dtype,
        X._shape, X._nshape,
        True,
        value_cache, 10
    )
