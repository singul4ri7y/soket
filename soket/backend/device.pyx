from cpython.object cimport Py_TYPE, PyTypeObject
from cpython.unicode cimport PyUnicode_AsUTF8, PyUnicode_DecodeUTF8
from libc.string cimport strcmp, memcpy
from libc.stdio cimport snprintf
from cython cimport freelist
cimport numpy
cimport cupy
import numpy
import warnings


# Try importing CuPy. If failed, GPU device is unavailable.
cdef bint _gpu_available = False
try:
    import cupy
    from cupy.cuda import Device as cupy_device

    _gpu_available = True
except:
    warnings.warn('CuPy is not available. GPU devices are disabled in Soket.')

# Returns true if GPU is available
cdef bint _is_gpu_available():
    return _gpu_available


## SUPPORTED DEVICES ##
DEF _NUM_SUPPORTED_DEVICES = 2
cdef (char *)[_NUM_SUPPORTED_DEVICES] _supported_devices = [ 'cpu', 'gpu' ]


@freelist(32)
cdef class Device:
    ''' Represents a compute backend device. '''

    def __cinit__(self, type: DeviceType | str, id: int):
        # Forward declaration for Python scoping
        cdef const char *dev_str

        if Py_TYPE(type) is <PyTypeObject *> DeviceType:
            self._dev_idx = type
        elif Py_TYPE(type) is <PyTypeObject *> str:
            dev_str = PyUnicode_AsUTF8(type)
            for i in range(_NUM_SUPPORTED_DEVICES):
                if not strcmp(_supported_devices[i], dev_str):
                    self._dev_idx = i
                    break

        # Set id
        self._id = id

        if self._dev_idx == DeviceType.GPU:
            if not _gpu_available:
                raise RuntimeError(f'Cannot create a GPU device. Check warnings.')

            # Set backend device and also keep handle to GPU/CUDA device
            self._backend = cupy
            self._backend_device = cupy_device(id)
        elif self._dev_idx == DeviceType.CPU:
            # Set NumPy as backend device
            self._backend = numpy

        # Do function interning to reduce overhead
        self._rand_fn = self._backend.random.uniform
        self._randn_fn = self._backend.random.normal
        self._binomial_fn = self._backend.random.binomial
        self._zeros_fn = self._backend.zeros
        self._ones_fn = self._backend.ones
        self._eye_fn = self._backend.eye
        self._empty_fn = self._backend.empty
        self._full_fn = self._backend.full

    ## DUNDER METHODS ##

    def __enter__(self) -> Device:
        ''' When enter `with` block. '''

        global _DEFAULT_DEVICE

        # Save and switch the default device
        self._previous_default_device = _DEFAULT_DEVICE
        _DEFAULT_DEVICE = self

        if self._backend_device is not None:
            self._backend_device.__enter__()

        return self

    def __exit__(self, exec_type, exec_value, traceback):
        ''' When exit a `with` block. '''

        global _DEFAULT_DEVICE

        # Switch back to previous default device
        _DEFAULT_DEVICE = self._previous_default_device

        if self._backend_device is not None:
            self._backend_device.__exit__(exec_type, exec_value, traceback)

        # Device does not handle any exception
        return False

    def __eq__(self, other: Device) -> bool:
        ''' Check whether two devices are equal. '''

        if Py_TYPE(other) is not <PyTypeObject *> Device:
            return False
        return self._eq(other)

    def __ne__(self, other: Device) -> bool:
        ''' Check whether two devices are not equal. '''

        if Py_TYPE(other) is not <PyTypeObject *> Device:
            return False
        return self._ne(other)

    def __str__(self) -> str:
        ''' String representation of a device. '''

        cdef char[64] repr
        cdef Py_ssize_t length

        if self._dev_idx == DeviceType.GPU:
            memcpy(repr, 'soket.Device(GPU, ', 18 * sizeof(char))
            length = 18 + snprintf(repr + 18, 45, '%d', self._id)
            repr[length] = ')'
            length += 1
        else:
            length = 17
            memcpy(repr, 'soket.Device(CPU)', length * sizeof(char))

        # Add termination character
        repr[length] = '\0'

        return PyUnicode_DecodeUTF8(repr, length, NULL)

    def __repr__(self) -> str:
        ''' Representation of a device. '''

        return self.__str__()

    ## DUNDER METHODS END ##

    ## CDEF METHODS ##

    cdef bint _eq(self, Device other):
        ''' Check whether two devices are equal (C only). '''

        # Python scope forward declaration
        cdef bint is_equal

        is_equal = self._dev_idx == other._dev_idx
        if self._backend_device is not None:
            is_equal &= <bint> self._backend_device.__eq__(other._backend_device)

        return is_equal

    cdef bint _ne(self, Device other):
        ''' Check whether two devices are not equal (C only). '''

        return not self._eq(other)

    ## CDEF METHODS END ##

    ## PROPERTIES ##

    @property
    def type(self) -> DeviceType:
        ''' Returns the device type. '''

        return <DeviceType> self._dev_idx

    @property
    def id(self) -> int:
        ''' Returns the device ID being used (GPU only). '''

        cdef int id = -1

        if self._dev_idx == DeviceType.GPU:
            id = self._id

        return id

    ## PROPERTIES END ##

    ## GPU SPECIFIC OPERATIONS ##

    cpdef void use(self):
        ''' Use current GPU device. '''

        if self._dev_idx == DeviceType.GPU:
            self._backend_device.use()

    cpdef void sync(self):
        ''' Synchronizes current thread to the device. '''

        if self._dev_idx == DeviceType.GPU:
            self._backend_device.synchronize()

    ## GPU SPECIFIC OPERATIONS END ##

    ## DEVICE SPECIFIC TENSOR CREATION OPS ##

    cdef object _rand(self, tuple shape, object low, object high, str dtype):
        '''
        Returns samples from an uniform distribution of interval [low, high).
        '''

        return self._rand_fn(low, high, shape).astype(dtype)

    cdef object _randn(self, tuple shape, object mean, object std, str dtype):
        '''
        Returns samples from a normal distribution of provided mean and variance.
        '''

        return self._randn_fn(mean, std, shape).astype(dtype)

    cdef object _randb(self, tuple shape, object p, str dtype):
        '''
        Creates and returns a binomial distribution of given probability and
        single trial per sample.
        '''

        return self._binomial_fn(1, p, shape).astype(dtype)

    cdef object _zeros(self, tuple shape, str dtype):
        ''' Return tensor/ndarray filled with zeros. '''

        return self._zeros_fn(shape, dtype)

    cdef object _ones(self, tuple shape, str dtype):
        ''' Returns tensor/ndarray filled with ones. '''

        return self._ones_fn(shape, dtype)

    cdef object _one_hot(self, object i, object num_classes, str dtype):
        ''' Returns a one-hot encoded tensor/ndarray. '''

        return self._eye_fn(num_classes, None, 0, dtype).__getitem__(i)

    cdef object _empty(self, tuple shape, str dtype):
        ''' Return an uninitialized tensor/ndarray. '''

        return self._empty_fn(shape, dtype)

    cdef object _full(self, tuple shape, object fill, str dtype):
        ''' Returns a tensor/ndarray filled with some value. '''

        return self._full_fn(shape, fill, dtype)

    ## DEVICE SPECIFIC TENSOR CREATION OPS END ##


## DEVICES ##
cdef Device _cpu_device = Device(DeviceType.CPU, -1)
cdef Device _gpu_device_default
if _gpu_available:
    _gpu_device_default = Device(DeviceType.GPU, 0)

# Default device
cdef Device _DEFAULT_DEVICE = _cpu_device


## DEVICE FN ##

cdef Device _default_device():
    ''' Returns the default device in use. '''
    return _DEFAULT_DEVICE

def cpu():
    ''' Returns the CPU device. '''
    return _cpu_device

def gpu(id: int = None):
    ''' Returns a GPU device if available. '''

    if not _gpu_available:
        raise RuntimeError(f'Cannot create a GPU device. Check warnings.')

    # Return default GPU device
    if id is None or id == 0:
        return _gpu_device_default

    return Device(DeviceType.GPU, id)

## DEVICE FN END ##
