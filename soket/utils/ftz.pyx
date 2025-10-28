from libc.stdint cimport uint32_t


cdef extern from '<immintrin.h>':
    uint32_t _mm_getcsr()
    void _mm_setcsr(uint32_t)


# Computing subnormal numbers are extremely slow in CPUs. The following function
# enables Flush-to-Zero (FTZ) and Denormals-are-Zeros bit in the Multimedia
# Extension Control/Status Register (MXCSR) for SEE (also affects AVX), which
# will give incentive to flush all the subnormals to zero.
#
# FTZ -> Subnormal/denormal results are flushed to zero.
# DAZ -> Subnormal/denormal input operands are flushed to zeros.
#
# Source: https://www.cita.utoronto.ca/~merz/intel_c10b/main_cls/mergedProjects/fpops_cls/common/fpops_set_ftz_daz.htm
def _flush_subnormals_to_zero():
    cdef uint32_t FTZ_BIT = 1 << 15
    cdef uint32_t DAZ_BIT = 1 << 6

    cdef uint32_t csr = _mm_getcsr()
    csr |= FTZ_BIT | DAZ_BIT
    _mm_setcsr(csr)
