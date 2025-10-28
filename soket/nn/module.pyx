from cpython.ref cimport Py_XDECREF
from cpython.object cimport Py_TYPE, PyTypeObject, PyObject_IsInstance
from cpython.list cimport PyList_GET_SIZE, PyList_GET_ITEM
from cpython.tuple cimport PyTuple_GET_ITEM, PyTuple_GET_SIZE
from cpython.dict cimport PyDict_New, PyDict_Next
from libc.stdlib cimport free
from collections.abc import Iterator


## HELPER FUNCTIONS ##

def _extract_modules_generator(object value):
    ''' Generator for all the child modules (including self). '''

    # Python scope forward declaration
    cdef Module module
    cdef PyObject *key
    cdef PyObject *dvalue
    cdef Py_ssize_t pos

    if PyObject_IsInstance(value, Module):
        module = <Module> value
        yield module

        # Iterate through storage for builtin modules
        for i in range(module._nstorage):
            yield from _extract_modules_generator(<object> module._storage[i])
        yield from _extract_modules_generator(module.__dict__)
    elif Py_TYPE(value) is <PyTypeObject *> dict:
        pos = 0
        while PyDict_Next(value, &pos, &key, &dvalue):
            yield from _extract_modules_generator(<object> dvalue)
    elif Py_TYPE(value) is <PyTypeObject *> list:
        for i in range(PyList_GET_SIZE(value)):
            yield from _extract_modules_generator(
                <object> PyList_GET_ITEM(value, i)
            )
    elif Py_TYPE(value) is <PyTypeObject *> tuple:
        for i in range(PyTuple_GET_SIZE(value)):
            yield from _extract_modules_generator(
                <object> PyTuple_GET_ITEM(value, i)
            )

def _extract_params_generator(object value):
    ''' Generator for all module parameters (including child modules). '''

    # Python scope forward declaration
    cdef Module module
    cdef PyObject *key
    cdef PyObject *dvalue
    cdef Py_ssize_t pos

    if PyObject_IsInstance(value, Tensor):
        yield <Tensor> value
    elif PyObject_IsInstance(value, Module):
        module = <Module> value

        for i in range(module._nstorage):
            yield from _extract_params_generator(<object> module._storage[i])
        yield from _extract_params_generator(module.__dict__)
    elif Py_TYPE(value) is <PyTypeObject *> dict:
        pos = 0
        while PyDict_Next(value, &pos, &key, &dvalue):
            yield from _extract_params_generator(<object> dvalue)
    elif Py_TYPE(value) is <PyTypeObject *> list:
        for i in range(PyList_GET_SIZE(value)):
            yield from _extract_params_generator(
                <object> PyList_GET_ITEM(value, i)
            )
    elif Py_TYPE(value) is <PyTypeObject *> tuple:
        for i in range(PyTuple_GET_SIZE(value)):
            yield from _extract_params_generator(
                <object> PyTuple_GET_ITEM(value, i)
            )

cdef void _set_train_mode(Module module, bint mode):
    ''' Helper to set training mode throughout children modules. '''

    module._training = mode
    for i in range(module._nstorage):
        if PyObject_IsInstance(<object> module._storage[i], Module):
            _set_train_mode(<Module> module._storage[i], mode)

    cdef PyObject *key
    cdef PyObject *value
    cdef Py_ssize_t pos = 0
    while PyDict_Next(module.__dict__, &pos, &key, &value):
        if PyObject_IsInstance(<object> value, Module):
            _set_train_mode(<Module> value, mode)

## HELPER FUNCTIONS END ##


cdef class Module:
    ''' Represents a Soket module. '''

    def __init__(self):
        # By default a module parameter is always training.
        self._training = True

        # Builtin modules initially don't hold any parameters/modules.
        self._storage = NULL
        self._nstorage = 0

        # Initialize `__dict__`
        self.__dict__ = PyDict_New()

        # Not a builtin module by default.
        self._builtin = False

    def __dealloc__(self):
        ''' Module destructor. '''

        # Decrease object references
        for i in range(self._nstorage):
            Py_XDECREF(self._storage[i])

        if self._storage != NULL:
            free(self._storage)

    ## PROPERTIES ##

    @property
    def training(self) -> bool:
        return self._training

    ## PROPERTIES END ##

    ## METHODS ##

    def modules(self) -> Iterator[Module]:
        ''' Returns a generator with child modules (including self). '''

        return _extract_modules_generator(self)

    def parameters(self) -> Iterator[Parameter]:
        ''' Returns a generator of list of parameters of of a module. '''

        return _extract_params_generator(self)

    def train(self, mode: bool = True) -> None:
        ''' Set the module and child modules in training mode. '''

        _set_train_mode(self, <bint> mode)

    def eval(self) -> None:
        ''' Set the module and child modules in evaluation mode. '''

        _set_train_mode(self, False)

    ## METHODS END ##

    ## CDEF METHODS ##

    cdef Tensor _fast_forward(self, Tensor X, object y):
        pass

    ## CDEF METHODS END ##

    ## DUNDER METHODS ##

    def __call__(self, *args):
        ''' Forward propagate when call the module. '''

        cdef Py_ssize_t len = PyTuple_GET_SIZE(args)
        if len == 0:
            raise ValueError('Expected atleast one positional argument!')

        # Fast builtin modules
        if self._builtin:
            return self._fast_forward(
                <Tensor> PyTuple_GET_ITEM(args, 0),
                None if len == 1 else <object> PyTuple_GET_ITEM(args, 1)
            )

        return self.forward(*args)

    def __str__(self) -> str:
        ''' Generic module representation in string '''

        if Py_TYPE(self) is <PyTypeObject *> Module:
            return 'soket.nn.Module()'
        return self.__class__.__name__ + '()'

    def __repr__(self) -> str:
        return self.__str__()

    ## DUNDER METHODS END ##
