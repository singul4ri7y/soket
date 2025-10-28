from cpython.object cimport Py_TYPE, PyTypeObject
from cpython.tuple cimport PyTuple_GET_SIZE, PyTuple_GET_ITEM
from soket.tensor cimport Tensor
import numpy

cdef class Transform:
    ''' Base class for transformation. '''

    cdef bint _builtin

    def __cinit__(self):
        # By default not a builtin function
        self._builtin = False

    ## METHODS ##

    def transform(self, *args):
        raise NotImplementedError()

    ## METHODS END ##

    ## DUNDER METHODS ##

    def __call__(self, *args):
        ''' Apply transformation when called. '''

        cdef Py_ssize_t len = PyTuple_GET_SIZE(args)
        if len == 0:
            raise ValueError('Expected atleast one positional argument!')

        if self._builtin:
            return self._fast_transform(<object> PyTuple_GET_ITEM(args, 0))

        return self.transform(*args)

    ## DUNDER METHODS END ##

    ## CDEF METHODS ##

    cdef object _fast_transform(self, object X):
        raise NotImplementedError()

    ## CDEF METHODS ##


# TODO: Convert to Cython
# class Compose(Transform):
#     """
#     Similar to nn.Sequential. Usage:
#         transforms.Compose(
#             transforms.ToTensor,
#             transforms.RandomCrop(padding=3)
#         )
#     """
#
#     def __init__(self, transforms: List[Transform] | Tuple[Transform]):
#         for t in transforms:
#             assert isinstance(t, Transform)
#
#         self.transforms = transforms
#
#     def transform(self, X):
#         for tform in self.transforms:
#             X = tform(X)
#
#         return X


# Interning
cdef object _ndarray = numpy.ndarray
cdef object _array = numpy.array

cdef class ToTensor(Transform):
    ''' Transforms a NumPy data sample to Tensor. '''

    def __cinit__(self):
        # Builtin transformer
        self._builtin = True

    ## CDEF METHODS ##

    cdef object _fast_transform(self, object X):
        # Convert the data sample to numpy.ndarray
        if Py_TYPE(X) is not <PyTypeObject *> _ndarray:
            X = _array(X)

        return Tensor._from_numpy(X)

    ## CDEF METHODS END ##
