from cpython.object cimport PyObject


cdef class Module:
    # Are parameters being trained?
    cdef bint _training

    # Parameters list for builtin modules
    cdef PyObject **_storage
    cdef int _nstorage
