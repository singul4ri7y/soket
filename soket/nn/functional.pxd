from soket.tensor cimport Tensor


cpdef Tensor batch_norm(
    Tensor X,
    Tensor running_mean,
    Tensor running_var,
    Tensor gamma=?,
    Tensor beta=?,
    object training=?,
    object momentum=?,
    object eps=?
)


cpdef Tensor layer_norm(
    Tensor X,
    Tensor weight=?,
    Tensor bias=?,
    object eps=?
)

