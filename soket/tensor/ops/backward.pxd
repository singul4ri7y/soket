from soket.tensor cimport Tensor, TensorTriad


cdef TensorTriad _elemwise_add_bwd(
    Tensor node,
    Tensor adj,
    Tensor x, Tensor y, Tensor Z
)

cdef TensorTriad _scalar_add_bwd(
    Tensor node,
    Tensor adj,
    Tensor x, Tensor y, Tensor Z
)


cdef TensorTriad _negate_bwd(
    Tensor node,
    Tensor adj,
    Tensor x, Tensor y, Tensor Z
)


cdef TensorTriad _elemwise_sub_bwd(
    Tensor node,
    Tensor adj,
    Tensor x, Tensor y, Tensor Z
)


cdef TensorTriad _scalar_sub_bwd(
    Tensor node,
    Tensor adj,
    Tensor x, Tensor y, Tensor Z
)


cdef TensorTriad _elemwise_mul_bwd(
    Tensor node,
    Tensor adj,
    Tensor x, Tensor y, Tensor Z
)


cdef TensorTriad _scalar_mul_bwd(
    Tensor node,
    Tensor adj,
    Tensor x, Tensor y, Tensor Z
)


cdef TensorTriad _elemwise_div_bwd(
    Tensor node,
    Tensor adj,
    Tensor x, Tensor y, Tensor Z
)


cdef TensorTriad _scalar_div_bwd(
    Tensor node,
    Tensor adj,
    Tensor x, Tensor y, Tensor Z
)


cdef TensorTriad _elemwise_pow_bwd(
    Tensor node,
    Tensor adj,
    Tensor x, Tensor y, Tensor Z
)


cdef TensorTriad _scalar_pow_bwd(
    Tensor node,
    Tensor adj,
    Tensor x, Tensor y, Tensor Z
)


cdef TensorTriad _broadcast_to_bwd(
    Tensor node,
    Tensor adj,
    Tensor x, Tensor y, Tensor Z
)


cdef TensorTriad _sum_bwd(
    Tensor node,
    Tensor adj,
    Tensor x, Tensor y, Tensor Z
)


cdef TensorTriad _mean_bwd(
    Tensor node,
    Tensor adj,
    Tensor x, Tensor y, Tensor Z
)


cdef TensorTriad _max_bwd(
    Tensor node,
    Tensor adj,
    Tensor x, Tensor y, Tensor Z
)


cdef TensorTriad _min_bwd(
    Tensor node,
    Tensor adj,
    Tensor x, Tensor y, Tensor Z
)


cdef TensorTriad _matmul_bwd(
    Tensor node,
    Tensor adj,
    Tensor x, Tensor y, Tensor Z
)


cdef TensorTriad _reshape_bwd(
    Tensor node,
    Tensor adj,
    Tensor x, Tensor y, Tensor Z
)


cdef TensorTriad _permute_bwd(
    Tensor node,
    Tensor adj,
    Tensor x, Tensor y, Tensor Z
)


cdef TensorTriad _transpose_bwd(
    Tensor node,
    Tensor adj,
    Tensor x, Tensor y, Tensor Z
)


cdef TensorTriad _select_bwd(
    Tensor node,
    Tensor adj,
    Tensor x, Tensor y, Tensor Z
)


cdef TensorTriad _relu_bwd(
    Tensor node,
    Tensor adj,
    Tensor x, Tensor y, Tensor Z
)


cdef TensorTriad _log_bwd(
    Tensor node,
    Tensor adj,
    Tensor x, Tensor y, Tensor Z
)


cdef TensorTriad _exp_bwd(
    Tensor node,
    Tensor adj,
    Tensor x, Tensor y, Tensor Z
)


cdef TensorTriad _logsumexp_bwd(
    Tensor node,
    Tensor adj,
    Tensor x, Tensor y, Tensor Z
)


cdef TensorTriad _sxentropyloss_bwd(
    Tensor node,
    Tensor adj,
    Tensor x, Tensor y, Tensor Z
)


cdef TensorTriad _bnorm_bwd(
    Tensor node,
    Tensor adj,
    Tensor x, Tensor y, Tensor Z
)
