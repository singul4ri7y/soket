cdef class DType:
    ''' Represents a Soket datatype. '''

    # Store dtype information
    cdef char *_name
    cdef Py_ssize_t _length

    # Store an index to help make type promotion faster
    cdef int _idx

    ## CDEF METHODS ##

    cdef bint _eq(self, DType other)
    cdef bint _ne(self, DType other)
    cdef str _str(self)

    ## CDEF METHODS END ##

# Default datatype
cdef DType _default_datatype

cdef DType _get_scalar_dtype(scalar: int | float | bool)
cpdef DType promote_types(DType type_a, DType type_b)