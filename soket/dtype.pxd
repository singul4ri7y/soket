cdef class DType:
    ''' Represents a Soket datatype. '''

    # Store dtype information
    cdef char *_name
    cdef Py_ssize_t _length

    # Store an index to help make type promotion faster
    cdef int _idx

    ## CDEF METHODS ##

    cdef str _str(self)

    ## CDEF METHODS END ##

## DATATYPES ##

cdef DType float16
cdef DType float32
cdef DType float64

cdef DType int8
cdef DType uint8
cdef DType int16
cdef DType uint16
cdef DType int32
cdef DType uint32
cdef DType int64
cdef DType uint64

cdef DType _bool

# Default datatype
cdef DType _default_datatype

## DATATYPES END ##

cdef DType _get_scalar_dtype(scalar: int | float | bool)
cpdef DType promote_types(DType type_a, DType type_b)
