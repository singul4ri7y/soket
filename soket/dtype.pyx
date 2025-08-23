from cpython.object cimport Py_TYPE, PyTypeObject
from cpython.unicode cimport (PyUnicode_AsUTF8AndSize, PyUnicode_AsUTF8,
    PyUnicode_DecodeUTF8)
from cython cimport freelist
from libc.string cimport strcmp, memcpy
cimport numpy as np
import numpy as np


# Supported datatypes
DEF _NUM_DTYPES = 12
cdef (char *)[_NUM_DTYPES] _supported_dtypes = [ 'float16', 'float32', 'float64',
    'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64',
    'bool' ]


@freelist(32)
cdef class DType:
    ''' Represents a Soket datatype. '''

    def __cinit__(self, type: str | DType):
        # Cython use Python scoping
        cdef DType other
        cdef bint found = False  # Supported datatype check loop
        cdef char *type_str

        if Py_TYPE(type) is <PyTypeObject *> DType:
            # Convert
            other = <DType> type

            self._length = other._length
            self._idx = other._idx

            # Steal the `_name` pointer
            self._name = other._name
        elif Py_TYPE(type) is <PyTypeObject *> str:
            type_str = <char *> PyUnicode_AsUTF8AndSize(type, &self._length)

            # Check if the datatype is supported
            for i in range(_NUM_DTYPES):
                if not strcmp(type_str, _supported_dtypes[i]):
                    # Found
                    found = True

                    # Store datatype pointer to current index
                    self._name = _supported_dtypes[i]

                    # Store the index to use in promotion matrix
                    self._idx = i
            
            if not found:
                raise ValueError(f"Unsupported datatype '{type}'")
        else:
            raise ValueError(f"Invalid argument '{type}'")

    ## PROPERTIES ##

    @property
    def name(self) -> str:
        ''' Name of the datatype in use. '''

        return self._str()

    ## PROPERTIES END ##

    ## CDEF METHODS ##

    cdef bint _eq(self, DType other):
        ''' Test equality (C only). '''
        
        return self._idx == other._idx

    cdef bint _ne(self, DType other):
        ''' Test inequality (C only). '''

        return not self._eq(other)
    
    cdef str _str(self):
        ''' Get string representation (C only). '''

        return PyUnicode_DecodeUTF8(self._name, self._length, NULL)

    ## CDEF METHODS END ##

    ## DUNDER METHODS ##

    def __str__(self) -> str:
        ''' Get string representation. '''

        return self._str()

    def __repr__(self) -> str:
        ''' Representation of Soket DType. '''

        cdef char[24] buff
        cdef Py_ssize_t length = 13 + self._length  # Approx. length

        memcpy(buff, 'soket.DType(', 12 * sizeof(buff[0]))
        memcpy(buff + 12, self._name, self._length * sizeof(buff[0]))

        # Closing bracket and termination character
        buff[length - 1] = ')'
        buff[length] = '\0'

        return PyUnicode_DecodeUTF8(buff, length, NULL)

    def __eq__(self, type: str | DType) -> bool:
        ''' Test equality. '''

        if Py_TYPE(type) is <PyTypeObject *> DType:
            return self._idx == (<DType> type)._idx
        elif Py_TYPE(type) is <PyTypeObject *> str:
            return not strcmp(self._name, PyUnicode_AsUTF8(type))
        else:
            return False

    def __ne__(self, type: str | DType) -> bool:
        ''' Test inequality. '''

        return not self.__eq__(type)

    ## DUNDER METHODS END ##


## DATATYPES ##

cdef DType float16 = DType('float16')
cdef DType float32 = DType('float32')
cdef DType float64 = DType('float64')

cdef DType int8   = DType('int8')
cdef DType uint8  = DType('uint8')
cdef DType int16  = DType('int16')
cdef DType uint16 = DType('uint16')
cdef DType int32  = DType('int32')
cdef DType uint32 = DType('uint32')
cdef DType int64  = DType('int64')
cdef DType uint64 = DType('uint64')

cdef DType _bool  = DType('bool')

# Default datatype for any type of tensor creation in Soket
cdef DType _default_datatype = float32

## DATATYPES END ##


# Get datatype representing each scalar
cdef DType _get_scalar_dtype(scalar: int | float | bool):
    # Supported scalars are int, float and bool.

    if Py_TYPE(scalar) is <PyTypeObject *> int:
        return int32
    elif Py_TYPE(scalar) is <PyTypeObject *> float:
        return float32
    elif Py_TYPE(scalar) is <PyTypeObject *> bool:
        return _bool
    
    raise ValueError(f"Unsupported scalar '{scalar}'")


## TYPE PROMOTION ##

# Promotion matrix excludes boolean
cdef DType[:, :] _promotion_matrix = np.array([
    [float16, float32, float64, float16, float16, float16, float16, float16, float16, float16, float16],
    [float32, float32, float64, float32, float32, float32, float32, float32, float32, float32, float32],
    [float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64],
    [float16, float32, float64, int8,    int16,   int16,   int32,   int32,   int64,   int64,   float32],
    [float16, float32, float64, int16,   uint8,   int16,   uint16,  int32,   uint32,  int64,   uint64],
    [float16, float32, float64, int16,   int16,   int16,   int32,   int32,   int64,   int64,   float32],
    [float16, float32, float64, int32,   uint16,  int32,   uint16,  int32,   uint32,  int64,   uint64],
    [float16, float32, float64, int32,   int32,   int32,   int32,   int32,   int64,   int64,   float32],
    [float16, float32, float64, int64,   uint32,  int64,   uint32,  int64,   uint32,  int64,   uint64],
    [float16, float32, float64, int64,   int64,   int64,   int64,   int64,   int64,   int64,   float32],
    [float16, float32, float64, float32, uint64,  float32, uint64,  float32, uint64, float32,  uint64]
], dtype=DType)


# Returns promoted type (if applicable) between two types
cpdef DType promote_types(DType type_a, DType type_b):
    cdef int a_id = type_a._idx
    cdef int b_id = type_b._idx

    # Do nothing if they are equal.
    if a_id == b_id:
        return type_a
    
    # Boolean holds least rank in dtype, thus should
    # always be casted to other provided dtype
    if a_id == _bool._idx: return type_b
    if b_id == _bool._idx: return type_a

    return _promotion_matrix[a_id][b_id]

## TYPE PROMOTION END ##