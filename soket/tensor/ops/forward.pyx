## FORWARD PASS OPERATIONS ##

from soket.tensor cimport Tensor

cdef object _elemwise_add_fwd(
    object target,

    object x,
    int *x_shape,
    int x_nshape,

    object y,
    int *y_shape,
    int y_nshape
):
    ''' Performs element-wise addition on the data. '''

    cdef Tensor ten = <Tensor> target
    return ten._device._backend.add(
        x, y,
        dtype=ten._dtype._str() if ten._dtype is not None else None
    )