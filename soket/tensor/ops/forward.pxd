cdef object _elemwise_add_fwd(
    object target,

    object x,
    int *x_shape,
    int x_nshape,

    object y,
    int *y_shape,
    int y_nshape
)