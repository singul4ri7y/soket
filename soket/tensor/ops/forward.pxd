from soket.tensor cimport Tensor


cdef object _elemwise_add_fwd(Tensor target, object x, object y)
cdef object _scalar_add_fwd(Tensor target, object x, object y)

cdef object _negate_fwd(Tensor target, object x, object y)

cdef object _elemwise_sub_fwd(Tensor target, object x, object y)
cdef object _scalar_sub_fwd(Tensor target, object x, object y)

cdef object _elemwise_mul_fwd(Tensor target, object x, object y)
cdef object _scalar_mul_fwd(Tensor target, object x, object y)

cdef object _elemwise_div_fwd(Tensor target, object x, object y)
cdef object _scalar_div_fwd(Tensor target, object x, object y)

cdef object _elemwise_pow_fwd(Tensor target, object x, object y)
cdef object _scalar_pow_fwd(Tensor target, object x, object y)

cdef object _broadcast_to_fwd(Tensor target, object x, object y)

cdef object _sum_fwd(Tensor target, object x, object y)

cdef object _mean_fwd(Tensor target, object x, object y)

cdef object _max_fwd(Tensor target, object x, object y)
cdef object _min_fwd(Tensor target, object x, object y)

cdef object _matmul_fwd(Tensor target, object x, object y)

cdef object _reshape_fwd(Tensor target, object x, object y)

cdef object _permute_fwd(Tensor target, object x, object y)

cdef object _transpose_fwd(Tensor target, object x, object y)

cdef object _select_fwd(Tensor target, object x, object y)
