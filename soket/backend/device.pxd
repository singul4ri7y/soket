# Checks and returns whether GPU is available.
cdef bint _is_gpu_available()


# Denotes a device type
cpdef enum DeviceType:
    CPU = 0
    GPU = 1


cdef class Device:
    ''' Represents a compute backend device. '''

    # Device details
    cdef int _dev_idx  # Device index in `_supported_devices`
    cdef int _id       # Ignored for CPU devices

    ## DEVICE INTERNALS ##

    # Using NumPy as CPU backend
    # Using CuPy as GPU backend
    cdef object _backend
    cdef object _backend_device

    # Intern the frequently called function to reduce overhead
    cdef object _rand_fn
    cdef object _randn_fn
    cdef object _binomial_fn
    cdef object _zeros_fn
    cdef object _ones_fn
    cdef object _eye_fn
    cdef object _empty_fn
    cdef object _full_fn

    ## DEVICE INTERNALS END ##

    # Store previous default device when entering a `with` block
    cdef Device _previous_default_device

    ## CDEF METHODS ##

    cdef bint _eq(self, Device other)
    cdef bint _ne(self, Device other)

    ## CDEF METHODS END ##

    ## GPU SPECIFIC OPERATIONS ##

    cpdef void use(self)
    cpdef void sync(self)

    ## GPU SPECIFIC OPERATIONS END ##

    ## DEVICE SPECIFIC TENSOR CREATION OPS ##

    cdef object _rand(self, tuple shape, object low, object high, str dtype)
    cdef object _randn(self, tuple shape, object mean, object std, str dtype)
    cdef object _randb(self, tuple shape, object p, str dtype)
    cdef object _zeros(self, tuple shape, str dtype)
    cdef object _ones(self, tuple shape, str dtype)
    cdef object _one_hot(self, object i, object num_classes, str dtype)
    cdef object _empty(self, tuple shape, str dtype)
    cdef object _full(self, tuple shape, object fill, str dtype)

    ## DEVICE SPECIFIC TENSOR CREATION OPS END ##

# Get default device
cdef Device _default_device()
