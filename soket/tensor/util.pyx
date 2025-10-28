from cpython.object cimport PyObject_IsInstance, Py_TYPE, PyTypeObject
from soket.tensor cimport Tensor
from soket.tensor.ops.intern cimport _stack
from soket.backend cimport Device
from collections.abc import Sequence


cpdef Tensor stack(object tensors, int axis=0):
    '''
    Stacks the tensors of given sequence on top of each other and returns
    the resulting tensor.

    Note: This operation does not support backward pass due to some design
          limitations.
    '''

    # Check whether input tensors are a sequence
    if not PyObject_IsInstance(tensors, Sequence):
        raise ValueError('Expected a sequence of tensors!')

    # Validate tensors
    cdef Device device = None
    cdef Tensor tensor
    for tensor in tensors:
        if Py_TYPE(tensor) is not <PyTypeObject *> Tensor:
            raise ValueError('Expected a sequence of tensors!')

        if device is None:
            device = tensor._device
        elif tensor._device._ne(device):
            raise ValueError(
                'Tensors should share the same device in the sequence!'
            )

    return Tensor(
        _stack(device._dev_idx)(
            [(<Tensor> x)._compute_data() for x in tensors],
            axis=axis
        )
    )

