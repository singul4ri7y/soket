## INTERNING FUNCTIONS BASED ON DEVICE REDUCING ATTRIBUTE FETCHING OVERHAD ##

from cpython.object cimport PyObject
from soket.backend cimport DeviceType


# Number of devices
DEF _NUM_SUPPORTED_DEVICES = 2


## ARRAY
cdef (PyObject *)[_NUM_SUPPORTED_DEVICES] _ARRAY
cdef object _array(int dev)


## ADDITION
cdef (PyObject *)[_NUM_SUPPORTED_DEVICES] _ADD
cdef object _add(int dev)


## ADDITION
cdef (PyObject *)[_NUM_SUPPORTED_DEVICES] _NEG
cdef object _neg(int dev)


## SUBTRACTION
cdef (PyObject *)[_NUM_SUPPORTED_DEVICES] _SUB
cdef object _sub(int dev)


## MULTIPLICATION
cdef (PyObject *)[_NUM_SUPPORTED_DEVICES] _MUL
cdef object _mul(int dev)


## DIVISION
cdef (PyObject *)[_NUM_SUPPORTED_DEVICES] _DIV
cdef object _div(int dev)


## POWER
cdef (PyObject *)[_NUM_SUPPORTED_DEVICES] _POW
cdef object _pow(int dev)


## SUM
cdef (PyObject *)[_NUM_SUPPORTED_DEVICES] _SUM
cdef object _sum(int dev)


## MEAN
cdef (PyObject *)[_NUM_SUPPORTED_DEVICES] _MEAN
cdef object _mean(int dev)


## MAX
cdef (PyObject *)[_NUM_SUPPORTED_DEVICES] _MAX
cdef object _max(int dev)


## MIN
cdef (PyObject *)[_NUM_SUPPORTED_DEVICES] _MIN
cdef object _min(int dev)


## ARGMAX
cdef (PyObject *)[_NUM_SUPPORTED_DEVICES] _ARGMAX
cdef object _argmax(int dev)


## ARGMIN
cdef (PyObject *)[_NUM_SUPPORTED_DEVICES] _ARGMIN
cdef object _argmin(int dev)


## RESHAPE
cdef (PyObject *)[_NUM_SUPPORTED_DEVICES] _RESHAPE
cdef object _reshape(int dev)


## BROADCAST
cdef (PyObject *)[_NUM_SUPPORTED_DEVICES] _BCASTTO
cdef object _broadcast_to(int dev)


## LOG
cdef (PyObject *)[_NUM_SUPPORTED_DEVICES] _LOG
cdef object _log(int dev)


## MATMUL
cdef (PyObject *)[_NUM_SUPPORTED_DEVICES] _MATMUL
cdef object _matmul(int dev)


## COPY
cdef (PyObject *)[_NUM_SUPPORTED_DEVICES] _COPY
cdef object _copy(int dev)


## EQUAL
cdef (PyObject *)[_NUM_SUPPORTED_DEVICES] _EQUAL
cdef object _equal(int dev)


## NOT EQUAL
cdef (PyObject *)[_NUM_SUPPORTED_DEVICES] _NOT_EQUAL
cdef object _not_equal(int dev)


## GREATER
cdef (PyObject *)[_NUM_SUPPORTED_DEVICES] _GREATER
cdef object _greater(int dev)


## GREATER EQUAL
cdef (PyObject *)[_NUM_SUPPORTED_DEVICES] _GREATER_EQUAL
cdef object _greater_equal(int dev)


## LESS
cdef (PyObject *)[_NUM_SUPPORTED_DEVICES] _LESS
cdef object _less(int dev)


## LESS EQUAL
cdef (PyObject *)[_NUM_SUPPORTED_DEVICES] _LESS_EQUAL
cdef object _less_equal(int dev)


## TRANSPOSE
cdef (PyObject *)[_NUM_SUPPORTED_DEVICES] _TRANSPOSE
cdef object _transpose(int dev)
