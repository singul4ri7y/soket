from cpython.object cimport PyObject
from soket.tensor cimport Tensor


cdef class Parameter(Tensor):
    ''' Denotes a parameter in modules. '''


cdef class Module:
    # Are parameters being trained?
    cdef bint _training

    # Parameters list for builtin modules
    cdef PyObject **_storage
    cdef int _nstorage

    # To keep consistency between builtin modules and Python modules
    # Never to be used by builtin modules
    cdef dict __dict__

    # If builtin module, calling `_fast_forward()` can reduce Python
    # calling overhead
    cdef bint _builtin

    ## CDEF METHODS ##

    cdef Tensor _fast_forward(self, Tensor X, object y)

    ## CDEF METHODS END ##
