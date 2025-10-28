from cpython.object cimport PyObject, Py_TYPE, PyTypeObject
from cpython.list cimport PyList_GET_SIZE, PyList_GET_ITEM
from cpython.tuple cimport PyTuple_GET_SIZE, PyTuple_GET_ITEM
from soket.tensor cimport Tensor
from soket.tensor.ops.intern cimport *
from libc.stdlib cimport malloc, free
cimport numpy as np
import numpy as np


cdef class Optimizer:
    ''' Optimizer base class. '''

    # Parameters list
    cdef PyObject **_params
    cdef int _param_size

    def __init__(self, object params):
        params = list(params)
        self._param_size = PyList_GET_SIZE(params)

        self._params = <PyObject **> malloc(
            self._param_size * sizeof(PyObject *)
        )
        if self._params == NULL:
            raise MemoryError('Failed to allocate buffer to store parameters')

        for i in range(self._param_size):
            self._params[i] = <PyObject *> PyList_GET_ITEM(params, i)

    def __dealloc__(self):
        ''' Release allocations. '''

        if self._params != NULL:
            free(self._params)

    def step(self):
        raise NotImplementedError()


# Stochastic variant of Gradient Descent with (nesterov) momentum.
cdef class SGD(Optimizer):
    ''' Stochastic Gradient Descent optimizer. '''

    # Optimizer hyper-parameters
    cdef object _lr
    cdef object _momentum
    cdef object _one_minus_dampening
    cdef object _weight_decay
    cdef bint _have_momentum
    cdef bint _have_weight_decay
    cdef bint _nesterov
    cdef bint _maximize

    # Momentum for each parameter
    cdef object[:] _u

    def __init__(
        self,
        params,
        object lr=0.01,
        object momentum=0.0,
        object dampening=0.0,
        object weight_decay=0.0,
        object nesterov=False,
        object maximize=False
    ):
        super().__init__(params)

        self._lr = lr
        self._momentum = momentum
        self._have_momentum = (momentum == 0.0) is True
        self._one_minus_dampening = 1.0 - dampening
        self._weight_decay = weight_decay
        self._have_weight_decay = (weight_decay != 0.0) is True
        self._nesterov = nesterov is True
        self._maximize = maximize is True

        # Create memoryview to store momentum per parameter
        self._u = np.full((self._param_size,), None, dtype=object)

    def step(self):
        ''' A single stochastic gradient descent optimizer step. '''

        # Python scope forward declaration
        cdef Tensor p
        cdef object p_data, u, grad

        # Device index
        cdef int didx = (<Tensor> self._params[0])._device._dev_idx

        cdef Py_ssize_t i
        for i in range(self._param_size):
            p = <Tensor> self._params[i]
            p_data = p._compute_data()
            u = self._u[i]

            if p._grad is None:
                continue
            grad = (<Tensor> p._grad)._compute_data()

            # Apply weight decay for L2 regularization
            if self._have_weight_decay:
                # grad = grad.data + self._weight_decay * p.data
                grad = _add(didx)(grad, _mul(didx)(p_data, self._weight_decay))

            # # Compute momentum
            if self._have_momentum:
                if u is None:
                    u = grad
                else:
                    # u = momentum * u + (1 - dampening) * gradient
                    u = _add(didx)(
                        _mul(didx)(u, self._momentum),
                        _mul(didx)(self._one_minus_dampening, grad)
                    )

                self._u[i] = u

                # Nesterov momentum
                if self._nesterov:
                    # grad = grad + momentum * u
                    grad = _add(didx)(grad, _mul(didx)(self._momentum, u))
                else:
                    grad = u

            if self._maximize:
                grad = _neg(didx)(grad)

            # No need to recompute tensor._shape as shape should be compatible
            p._data_ = _sub(didx)(p_data, _mul(didx)(self._lr, grad))


cdef class Adam(Optimizer):
    ''' The Adam optimizer. '''

    # Optimizer hyper-parameters
    cdef object _lr
    cdef object _beta1
    cdef object _beta2
    cdef object _one_minus_beta1
    cdef object _one_minus_beta2
    cdef object _eps
    cdef object _weight_decay
    cdef bint _have_weight_decay
    cdef bint _maximize

    # Iteration tracking
    cdef object _t
    cdef object _beta1_t
    cdef object _beta2_t
    cdef object _one_minus_beta1_t
    cdef object _one_minus_beta2_t

    # Momentum tracking
    cdef object[:] _u
    cdef object[:] _v

    def __init__(
        self,
        params,
        object lr=0.001,
        object betas=(0.9, 0.999),
        object eps=1e-8,
        object weight_decay=0,
        object maximize=False
    ):
        super().__init__(params)

        cdef Py_ssize_t tuple_siz = PyTuple_GET_SIZE(betas)
        if tuple_siz < 2:
            raise ValueError('Invalid betas!')

        cdef object beta
        for i in range(tuple_siz):
            beta = <object> PyTuple_GET_ITEM(betas, i)
            if Py_TYPE(beta) is not <PyTypeObject *> float:
                raise ValueError('Betas must be floats!')

        self._lr = lr
        self._beta1 = <object> PyTuple_GET_ITEM(betas, 0)
        self._beta2 = <object> PyTuple_GET_ITEM(betas, 1)
        self._one_minus_beta1 = 1.0 - self._beta1
        self._one_minus_beta2 = 1.0 - self._beta2
        self._eps = eps
        self._weight_decay = weight_decay
        self._have_weight_decay = (weight_decay != 0.0) is True
        self._maximize = maximize is True

        # Iteration
        self._t = 1
        self._beta1_t = self._beta1
        self._beta2_t = self._beta2
        self._one_minus_beta1_t = 1.0 - self._beta1_t
        self._one_minus_beta2_t = 1.0 - self._beta2_t

        # Memoryview for momentums
        self._u = np.full((self._param_size,), None, dtype=object)
        self._v = np.full((self._param_size,), None, dtype=object)

    def step(self):
        # Python scope forward declaration
        cdef Tensor p
        cdef object p_data, u, v, grad

        # Device index
        cdef int didx = (<Tensor> self._params[0])._device._dev_idx

        for i in range(self._param_size):
            p = <Tensor> self._params[i]
            p_data = p._compute_data()
            u = self._u[i]
            v = self._v[i]

            if p._grad is None:
                continue
            grad = (<Tensor> p._grad)._compute_data()

            # Apply weight decay for L2 regularization
            if self._have_weight_decay:
                # grad = grad.data + self._weight_decay * p.data
                grad = _add(didx)(grad, _mul(didx)(p_data, self._weight_decay))

            if u is None:
                # u = grad * (1 - beta1)
                u = _mul(didx)(grad, self._one_minus_beta1)
            else:
                # u = beta1 * u + grad * (1 - beta1)
                u = _add(didx)(
                    _mul(didx)(u, self._beta1),
                    _mul(didx)(grad, self._one_minus_beta1)
                )
            self._u[i] = u

            if v is None:
                # v = grad * grad * (1 - beta2)
                v = _mul(didx)(grad, _mul(didx)(grad, self._one_minus_beta2))
            else:
                # v = beta2 * v + grad * grad * (1 - beta2)
                v = _add(didx)(
                    _mul(didx)(v, self._beta2),
                    _mul(didx)(grad, _mul(didx)(grad, self._one_minus_beta2))
                )
            self._v[i] = v

            # Biasing
            u = _div(didx)(u, self._one_minus_beta1_t)
            v = _div(didx)(v, self._one_minus_beta2_t)

            if self._maximize:
                grad = _neg(didx)(grad)

            # Update
            p._data_ = _sub(didx)(
                p_data,
                _mul(didx)(
                    self._lr,
                    _div(didx)(
                        u,
                        _add(didx)(_pow(didx)(v, 0.5), self._eps)
                    )
                )
            )

        self._t += 1
        self._beta1_t *= self._beta1
        self._beta2_t *= self._beta2
        self._one_minus_beta1_t = 1.0 - self._beta1_t
        self._one_minus_beta2_t = 1.0 - self._beta2_t
