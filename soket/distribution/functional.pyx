from cpython.object cimport PyObject, Py_TYPE, PyTypeObject
from cpython.long cimport PyLong_AsLong
from soket.backend cimport DeviceType, _is_gpu_available
from soket.tensor.ops.intern cimport *
from soket.tensor cimport Tensor
from soket.dtype cimport int64


def multinomial(
    prob: Tensor,
    num_samples: int,
    replacement=False,
    generator=None
):
    '''
    Returns a tensor with samples from the multinomial distribution of
    given probability.
    '''

    # Check for tensor input
    if Py_TYPE(prob) is not <PyTypeObject *> Tensor:
        raise ValueError('Expected the probability input to be tensor!')

    cdef Tensor p = <Tensor> prob
    # Upto 2D tensor is acceptable.
    if p._nshape > 2:
        raise ValueError('Probability distribution must be 1D/2D!')

    # Number of samples should be sufficient
    cdef int inum_samples = <int> PyLong_AsLong(num_samples)
    if (replacement is False and inum_samples >
    (p._shape[1] if p._nshape > 1 else p._shape[0])):
        raise ValueError('Number of samples cannot exceed dim[-1]!')

    cdef int didx = prob._device._dev_idx
    cdef object prob_data = p._compute_data()
    cdef object p_data

    # Create shape
    cdef int[2] shape
    cdef int nshape

    cdef object normalized_prob_data
    cdef object size
    cdef object oi
    if p._nshape == 1:
        normalized_prob_data = _div(didx)(
            prob_data,
            _sum(didx)(prob_data, None, None, None, True)
        )

        p_data = _random_choice(didx)(
            prob_data.size,
            num_samples,
            replacement,
            normalized_prob_data
        )

        # Prepare shape
        shape[0] = inum_samples
        nshape = 1
    else:
        normalized_prob_data = _div(didx)(
            prob_data,
            _sum(didx)(prob_data, (-1,), None, None, True)
        )

        size = p._shape[1]
        p_data = p._device._zeros((p._shape[0], num_samples), int64._str())
        for i in range(p._shape[0]):
            oi = i
            p_data.__setitem__(oi, _random_choice(didx)(
                size,
                num_samples,
                replacement,
                normalized_prob_data.__getitem__(oi)
            ))

        # Prepare shape
        shape[0] = p._shape[0]
        shape[1] = inum_samples
        nshape = 2

    return Tensor._make_const(
        p_data,
        p._device,
        int64,
        shape, nshape,
        True,
        False
    )
