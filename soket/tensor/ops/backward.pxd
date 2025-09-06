from soket.tensor cimport Tensor, BackwardOutput


cdef BackwardOutput _elemwise_add_bwd(
    Tensor node,
    Tensor adj,
    Tensor x, Tensor y
)

cdef BackwardOutput _scalar_add_bwd(
    Tensor node,
    Tensor adj,
    Tensor x, Tensor y
)


cdef BackwardOutput _negate_bwd(
    Tensor node,
    Tensor adj,
    Tensor x, Tensor y
)


cdef BackwardOutput _elemwise_sub_bwd(
    Tensor node,
    Tensor adj,
    Tensor x, Tensor y
)


cdef BackwardOutput _scalar_sub_bwd(
    Tensor node,
    Tensor adj,
    Tensor x, Tensor y
)


cdef BackwardOutput _elemwise_mul_bwd(
    Tensor node,
    Tensor adj,
    Tensor x, Tensor y
)


cdef BackwardOutput _scalar_mul_bwd(
    Tensor node,
    Tensor adj,
    Tensor x, Tensor y
)


cdef BackwardOutput _elemwise_div_bwd(
    Tensor node,
    Tensor adj,
    Tensor x, Tensor y
)


cdef BackwardOutput _scalar_div_bwd(
    Tensor node,
    Tensor adj,
    Tensor x, Tensor y
)


cdef BackwardOutput _elemwise_pow_bwd(
    Tensor node,
    Tensor adj,
    Tensor x, Tensor y
)


cdef BackwardOutput _scalar_pow_bwd(
    Tensor node,
    Tensor adj,
    Tensor x, Tensor y
)


cdef BackwardOutput _broadcast_to_bwd(
    Tensor node,
    Tensor adj,
    Tensor x, Tensor y
)


cdef BackwardOutput _sum_bwd(
    Tensor node,
    Tensor adj,
    Tensor x, Tensor y
)


cdef BackwardOutput _mean_bwd(
    Tensor node,
    Tensor adj,
    Tensor x, Tensor y
)


cdef BackwardOutput _max_bwd(
    Tensor node,
    Tensor adj,
    Tensor x, Tensor y
)


cdef BackwardOutput _min_bwd(
    Tensor node,
    Tensor adj,
    Tensor x, Tensor y
)


cdef BackwardOutput _matmul_bwd(
    Tensor node,
    Tensor adj,
    Tensor x, Tensor y
)


cdef BackwardOutput _reshape_bwd(
    Tensor node,
    Tensor adj,
    Tensor x, Tensor y
)


cdef BackwardOutput _permute_bwd(
    Tensor node,
    Tensor adj,
    Tensor x, Tensor y
)


cdef BackwardOutput _transpose_bwd(
    Tensor node,
    Tensor adj,
    Tensor x, Tensor y
)


cdef BackwardOutput _select_bwd(
    Tensor node,
    Tensor adj,
    Tensor x, Tensor y
)
