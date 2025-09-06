from cpython.ref cimport Py_XDECREF
from cpython.object cimport PyObject_IsInstance
from cpython.list cimport PyList_Append
from libc.stdlib cimport free


cdef list _extract_modules(object value, list acc):
    # Python scope forward declaration
    cdef Module module

    if PyObject_IsInstance(value, Module):
        PyList_Append(acc, value)

        module = <Module> value

        # Iterate through storage for builtin modules
        for i in range(module._nstorage):
            if PyObject_IsInstance(<object> module._storage[i], Module):
                PyList_Append(acc, <object> module._storage[i])


cdef class Module:
    def __cinit__(self):
        # By default a module parameter is always training.
        self._training = True

        # Builtin modules initially don't hold any parameters/modules.
        self._storage = NULL
        self._nstorage = 0

    def __del__(self):
        ''' Module destructor. '''

        # Decrease object references
        for i in range(self._nstorage):
            Py_XDECREF(self._storage[i])

        if self._storage != NULL:
            free(self._storage)


