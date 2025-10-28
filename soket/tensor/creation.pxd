from soket.tensor.tensor cimport Tensor
from soket.backend cimport Device
from soket.dtype cimport DType


cdef Tensor _randb(tuple shape, p, Device device, DType dtype, requires_grad)
cdef Tensor _zeros(tuple shape, Device device, DType dtype, requires_grad)
cdef Tensor _ones(tuple shape, Device device, DType dtype, requires_grad)


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


cdef Tensor _one_hot(
    Tensor tensor,
    int num_classes,
    Device device,
    DType dtype,
    requires_grad
)
