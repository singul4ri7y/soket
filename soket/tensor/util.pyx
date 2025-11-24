from cpython.object cimport PyObject_IsInstance, Py_TYPE, PyTypeObject
from cpython.tuple cimport PyTuple_GET_ITEM, PyTuple_Pack
from soket.tensor cimport Tensor
from soket.tensor.ops.intern cimport *
from soket.backend cimport Device
from collections.abc import Sequence


## HELPER FUNCTIONS ##

cdef Device _check_and_validate_tensor_sequence(object tensors):
    ''' Validate a tensor sequence and return the common device. '''

    # Check whether input tensors are a sequence
    if not PyObject_IsInstance(tensors, Sequence):
        raise ValueError('Expected a sequence of tensors!')

    # Validate tensors
    cdef Device device = None
    cdef Tensor ten
    for tensor in tensors:
        if Py_TYPE(tensor) is not <PyTypeObject *> Tensor:
            raise ValueError('Expected a sequence of tensors!')

        ten = <Tensor> tensor
        if device is None:
            device = ten._device
        elif ten._device._ne(device):
            raise ValueError(
                'Tensors should share the same device in the sequence!'
            )

    return device

## HELPER FUNCTIONS END ##


cpdef Tensor stack(object tensors, int axis=0):
    '''
    Stacks the tensors of given sequence on top of each other and returns
    the resulting tensor.

    Note: This operation does not support backward pass due to some design
          limitations.
    '''

    cdef Device dev = _check_and_validate_tensor_sequence(tensors)
    return Tensor(
        _stack(dev._dev_idx)(
            [(<Tensor> x)._compute_data() for x in tensors],
            axis=axis
        )
    )


def concat(object tensors, int axis=0):
    '''
    Performs tensor concatenation.

    Note: Concatenation operation does not support backward pass in Soket.
    '''

    cdef Device dev = _check_and_validate_tensor_sequence(tensors)
    return Tensor(
        _concat(dev._dev_idx)(
            [(<Tensor> x)._compute_data() for x in tensors],
            axis=axis
        )
    )


def histogram(
    Tensor input,
    bins=10,
    range=None,
    density=False,
    weights=None
) -> tuple[Tensor]:
    ''' Computes histogram of the values in a tensor. '''

    if Py_TYPE(bins) is <PyTypeObject *> Tensor:
        bins = (<Tensor> bins)._compute_data()

    cdef tuple hist = _hist(input._device._dev_idx)(
        input._compute_data(),
        bins,
        range, density, weights
    )

    cdef Tensor x = Tensor(<object> PyTuple_GET_ITEM(hist, 0))
    cdef Tensor y = Tensor(<object> PyTuple_GET_ITEM(hist, 1))
    return PyTuple_Pack(2, <PyObject *> x, <PyObject *> y)

