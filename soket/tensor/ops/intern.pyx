from soket.backend cimport _is_gpu_available
cimport numpy as np
cimport cupy as cp
import numpy as np


# Indicies
cdef int _cpu = DeviceType.CPU
cdef int _gpu = DeviceType.GPU


# Intern NumPy functions
_ARRAY[_cpu] = <PyObject *> np.array
_ADD[_cpu] = <PyObject *> np.add
_NEG[_cpu] = <PyObject *> np.negative
_SUB[_cpu] = <PyObject *> np.subtract
_MUL[_cpu] = <PyObject *> np.multiply
_DIV[_cpu] = <PyObject *> np.divide
_POW[_cpu] = <PyObject *> np.power
_SUM[_cpu] = <PyObject *> np.sum
_MEAN[_cpu] = <PyObject *> np.mean
_MAX[_cpu] = <PyObject *> np.max
_MIN[_cpu] = <PyObject *> np.min
_ARGMAX[_cpu] = <PyObject *> np.argmax
_ARGMIN[_cpu] = <PyObject *> np.argmin
_RESHAPE[_cpu] = <PyObject *> np.reshape
_BCASTTO[_cpu] = <PyObject *> np.broadcast_to
_LOG[_cpu] = <PyObject *> np.log
_EXP[_cpu] = <PyObject *> np.exp
_MATMUL[_cpu] = <PyObject *> np.matmul
_COPY[_cpu] = <PyObject *> np.copy
_EQUAL[_cpu] = <PyObject *> np.equal
_NOT_EQUAL[_cpu] = <PyObject *> np.not_equal
_GREATER[_cpu] = <PyObject *> np.greater
_GREATER_EQUAL[_cpu] = <PyObject *> np.greater_equal
_LESS[_cpu] = <PyObject *> np.less
_LESS_EQUAL[_cpu] = <PyObject *> np.less_equal
_TRANSPOSE[_cpu] = <PyObject *> np.transpose
_MAXIMUM[_cpu] = <PyObject *> np.maximum
_SQUEEZE[_cpu] = <PyObject *> np.squeeze
_STACK[_cpu] = <PyObject *> np.stack


# Intern GPU functions if available.
if _is_gpu_available():
    import cupy as cp

    _ARRAY[_gpu] = <PyObject *> cp.array
    _ADD[_gpu] = <PyObject *> cp.add
    _NEG[_gpu] = <PyObject *> cp.negative
    _SUB[_gpu] = <PyObject *> cp.subtract
    _MUL[_gpu] = <PyObject *> cp.multiply
    _DIV[_gpu] = <PyObject *> cp.divide
    _POW[_gpu] = <PyObject *> cp.power
    _SUM[_gpu] = <PyObject *> cp.sum
    _MEAN[_gpu] = <PyObject *> cp.mean
    _MAX[_gpu] = <PyObject *> cp.max
    _MIN[_gpu] = <PyObject *> cp.min
    _ARGMAX[_gpu] = <PyObject *> cp.argmax
    _ARGMIN[_gpu] = <PyObject *> cp.argmin
    _RESHAPE[_gpu] = <PyObject *> cp.reshape
    _BCASTTO[_gpu] = <PyObject *> cp.broadcast_to
    _LOG[_gpu] = <PyObject *> cp.log
    _EXP[_gpu] = <PyObject *> cp.exp
    _MATMUL[_gpu] = <PyObject *> cp.matmul
    _COPY[_gpu] = <PyObject *> cp.copy
    _EQUAL[_gpu] = <PyObject *> cp.equal
    _NOT_EQUAL[_gpu] = <PyObject *> cp.not_equal
    _GREATER[_gpu] = <PyObject *> cp.greater
    _GREATER_EQUAL[_gpu] = <PyObject *> cp.greater_equal
    _LESS[_gpu] = <PyObject *> cp.less
    _LESS_EQUAL[_gpu] = <PyObject *> cp.less_equal
    _TRANSPOSE[_gpu] = <PyObject *> cp.transpose
    _MAXIMUM[_gpu] = <PyObject *> cp.maximum
    _SQUEEZE[_gpu] = <PyObject *> cp.squeeze
    _STACK[_gpu] = <PyObject *> cp.stack


## INTERN HELPER FUNCTIONS ##

cdef object _array(int dev):
    return <object> _ARRAY[dev]

cdef object _add(int dev):
    return <object> _ADD[dev]

cdef object _neg(int dev):
    return <object> _NEG[dev]

cdef object _sub(int dev):
    return <object> _SUB[dev]

cdef object _mul(int dev):
    return <object> _MUL[dev]

cdef object _div(int dev):
    return <object> _DIV[dev]

cdef object _pow(int dev):
    return <object> _POW[dev]

cdef object _sum(int dev):
    return <object> _SUM[dev]

cdef object _mean(int dev):
    return <object> _MEAN[dev]

cdef object _max(int dev):
    return <object> _MAX[dev]

cdef object _min(int dev):
    return <object> _MIN[dev]

cdef object _argmax(int dev):
    return <object> _ARGMAX[dev]

cdef object _argmin(int dev):
    return <object> _ARGMIN[dev]

cdef object _reshape(int dev):
    return <object> _RESHAPE[dev]

cdef object _broadcast_to(int dev):
    return <object> _BCASTTO[dev]

cdef object _log(int dev):
    return <object> _LOG[dev]

cdef object _exp(int dev):
    return <object> _EXP[dev]

cdef object _matmul(int dev):
    return <object> _MATMUL[dev]

cdef object _copy(int dev):
    return <object> _COPY[dev]

cdef object _equal(int dev):
    return <object> _EQUAL[dev]

cdef object _not_equal(int dev):
    return <object> _NOT_EQUAL[dev]

cdef object _greater(int dev):
    return <object> _GREATER[dev]

cdef object _greater_equal(int dev):
    return <object> _GREATER_EQUAL[dev]

cdef object _less(int dev):
    return <object> _LESS[dev]

cdef object _less_equal(int dev):
    return <object> _LESS_EQUAL[dev]

cdef object _transpose(int dev):
    return <object> _TRANSPOSE[dev]

cdef object _maximum(int dev):
    return <object> _MAXIMUM[dev]

cdef object _squeeze(int dev):
    return <object> _SQUEEZE[dev]

cdef object _stack(int dev):
    return <object> _STACK[dev]

## INTERN HELPER FUNCTIONS END ##
