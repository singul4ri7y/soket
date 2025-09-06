from soket.tensor.tensor cimport Tensor
from soket.backend cimport Device
from soket.dtype cimport DType


cdef Tensor _zeros_like(
    Tensor ten,
    Device device,
    DType dtype,
    bint requires_grad
)


cdef Tensor _ones_like(
    Tensor ten,
    Device device,
    DType dtype,
    bint requires_grad
)
