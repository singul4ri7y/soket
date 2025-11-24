from cpython.object cimport PyObject
from cpython.long cimport PyLong_FromLong
from cpython.tuple cimport PyTuple_New, PyTuple_GET_ITEM, PyTuple_SET_ITEM
from soket.dtype cimport DType, int32, int64
from soket.tensor.creation cimport _one_hot
from soket.tensor cimport Op, OpType, TensorTriad
from soket.tensor.ops cimport *
from soket.tensor.ops.intern cimport *
from libc.stdlib cimport malloc
from libc.string cimport memcmp


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
    ''' This batchnorm version uses biased variance. '''
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
    ''' Biased estimator is used to calculate the variance. '''
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


cdef tuple sxe_axis_zero = (0,)
cdef tuple sxe_axis_one = (1,)

cpdef Tensor softmax_cross_entropy(
    Tensor logits, Tensor targets,
    str reduction='mean'
):
    '''
    Negative log-likelyhood loss on a softmax probability distributtion. Also
    known as the Softmax Cross-Entropy loss.
    '''

    if logits._device._ne(targets._device):
        raise RuntimeError('Incompatible device of classes and targets!')

    if logits._nshape == 0:
        raise ValueError('Expected the logits tensor to be atleast 1D!')

    # Check for valid reduction type
    cdef int batch_reduction = (reduction != 'none')
    if reduction != 'mean' and reduction != 'sum' and batch_reduction:
        raise ValueError(f'Invalid reduction type - {reduction}')

    # Check for shape compatibility.
    # w/o batch
    if logits._nshape == 1:
        if targets._nshape >= 0:
            raise ValueError(
                'Incompatible targets tensor, expected shape to be ()'
            )
    # with batch
    elif logits._nshape >= 2:
        if (targets._nshape + 1 != logits._nshape or
        logits._shape[0] != targets._shape[0] or
        # test channels
        memcmp(logits._shape + 2, targets._shape + 1, (logits._nshape - 2) *
        sizeof(int))):
            raise ValueError(
                f'Incompatible targets tensor shape - '
                f'{logits.shape} and {targets.shape}'
            )

    # LogSumExp reduction axes
    cdef tuple axes = sxe_axis_one if logits._nshape >= 2 else sxe_axis_zero

    # LogSumExp reduce axis (index)
    cdef int r_axis = 0 + logits._nshape >= 2
    # One hot encoded target
    cdef int num_classes = logits._shape[r_axis]
    cdef object one_hot = _one_hot(
        targets,
        num_classes,
        logits._device,
        logits._dtype,
        False
    )._compute_data()

    # If dealing with extra channels
    cdef tuple permute_axis
    cdef Py_ssize_t i
    cdef int idx
    if targets._nshape > 2:
        # TODO: Performance hit is pretty bad. FIXME.
        permute_axis = PyTuple_New(logits._nshape)
        PyTuple_SET_ITEM(permute_axis, 0, 0)

        # One hot encoded dimension should be right beside the batch
        PyTuple_SET_ITEM(permute_axis, 1, logits._nshape - 1)

        idx = 2
        for i in range(1, logits._nshape - 1):
            PyTuple_SET_ITEM(permute_axis, idx, i)
            idx += 1

        # Permute shape
        one_hot = _transpose(logits._device._dev_idx)(one_hot, permute_axis)

    # Calculate new tensor shape.
    cdef int *shape
    cdef int nshape

    if logits._nshape == 1:
        # scalar
        shape = <int *> malloc(0)
        nshape = 0
    else:
        nshape = logits._nshape - 1 - batch_reduction
        shape = <int *> malloc(nshape * sizeof(int))
        if shape == NULL:
            raise MemoryError('Failed to allocate shape for cross-entropy loss!')

        idx = 0
        for i in range(batch_reduction, logits._nshape):
            if i == r_axis:
                continue

            shape[idx] = logits._shape[i]
            idx += 1

    cdef (PyObject *)[4] value_cache = [
        <PyObject *> axes,
        <PyObject *> <object> False,
        <PyObject *> one_hot,
        <PyObject *> reduction,
    ]

    return Tensor._make_from_op(
        Op(
            _sxentropyloss_fwd,
            _sxentropyloss_bwd,
            OpType.SXENTROPYLOSS
        ),
        TensorTriad(<PyObject *> logits, NULL, NULL),
        logits._device,
        logits._dtype,
        shape, nshape,
        False,
        value_cache, 4
    )


cpdef Tensor embedding(Tensor input, Tensor weight):
    ''' Returns embedding vectors from embedding lookup table. '''

    if weight._nshape < 2:
        raise ValueError("'weight' must be 2D!")

    cdef DType dt = input._dtype
    if dt._idx != int32._idx and dt._idx != int64._idx:  # int32 or int64
        raise ValueError('Expected the input tensor be of type int32/int64!')

    return weight[input]
