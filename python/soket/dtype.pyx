from cpython.object cimport Py_TYPE, PyTypeObject, PyObject_Hash
from cpython.unicode cimport PyUnicode_AsUTF8AndSize, PyUnicode_AsUTF8,    \
    PyUnicode_DecodeUTF8
from libc.string cimport strncpy, strcmp, memcpy


# Supported datatypes
DEF _NUM_DTYPES = 12
cdef (char *)[_NUM_DTYPES] _supported_dtypes = [ 'float16', 'float32', 'float64',
    'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64',
    'bool' ]


cdef class DType:
    # Store dtype information
    cdef char[8] __name
    cdef Py_ssize_t __length

    # Hash of `__name`
    cdef Py_hash_t __hash

    def __cinit__(self, type: str | DType):
        # Cython use Python scoping
        cdef DType other

        if Py_TYPE(type) is <PyTypeObject *> DType:
            # Convert to raw DType class
            other = type

            self.__length = other.__length
            self.__hash = other.__hash

            strncpy(self.__name, other.__name, other.__length * sizeof(char))
        elif Py_TYPE(type) is <PyTypeObject *> str:
            strncpy(
                self.__name,
                PyUnicode_AsUTF8AndSize(type, &self.__length),
                self.__length
            )

            # Compute hash
            self.__hash = PyObject_Hash(
                PyUnicode_DecodeUTF8(self.__name, self.__length, NULL)
            )
        else:
            raise ValueError(f"Invalid argument '{type}'")

        cdef bint found = False
        for i in range(_NUM_DTYPES):
            if not strcmp(self.__name, _supported_dtypes[i]):
                found = True
                break

        if not found:
            raise ValueError(f"Unsupported datatype '{self.name}'")

    ## PROPERTIES ##

    @property
    def name(self) -> str:
        ''' Name of the datatype in use. '''

        return PyUnicode_DecodeUTF8(self.__name, self.__length, NULL)

    ## PROPERTIES END ##

    ## DUNDER METHODS ##

    def __str__(self) -> str:
        ''' Get string representation. '''

        return PyUnicode_DecodeUTF8(self.__name, self.__length, NULL)

    def __repr__(self) -> str:
        ''' Representation of Soket DType. '''

        cdef char[24] buff
        cdef Py_ssize_t length = 13 + self.__length  # Approx. length

        memcpy(buff, 'soket.DType(', 12 * sizeof(buff[0]))
        memcpy(buff + 12, self.__name, self.__length * sizeof(buff[0]))

        # Closing bracket and termination character
        buff[length - 1] = ')'
        buff[length] = '\0'

        return PyUnicode_DecodeUTF8(buff, length, NULL)

    def __hash__(self) -> int:
        ''' Return hash of datatype name string. '''

        return self.__hash

    def __eq__(self, type: str | DType) -> bool:
        ''' Test equality. '''

        # Forward declaration for Python scoping
        cdef DType other

        if Py_TYPE(type) is <PyTypeObject *> DType:
            other = type
            return self.__hash == other.__hash
        elif Py_TYPE(type) is <PyTypeObject *> str:
            return not strcmp(self.__name, PyUnicode_AsUTF8(type))
        else:
            return False

    def __ne__(self, type: str | DType) -> bool:
        ''' Test inequality. '''

        return not self.__eq__(type)

    ## DUNDER METHODS END ##

