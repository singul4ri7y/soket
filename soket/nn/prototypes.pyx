## DNN MODULE PROTOTYPES ##

from cpython.ref cimport Py_XINCREF
from cpython.object cimport Py_TYPE, PyTypeObject, PyObject, PyObject_IsInstance
from cpython.tuple cimport (PyTuple_GET_SIZE, PyTuple_GET_ITEM, PyTuple_New,
    PyTuple_SET_ITEM)
from cpython.list cimport PyList_GET_SIZE
from soket.nn.module cimport Module
from soket.tensor cimport Tensor, Op, OpType, TensorTriad
from soket.tensor.creation cimport _randb, _ones, _zeros, _one_hot
from soket.tensor.ops cimport (_relu_fwd, _relu_bwd, _sxentropyloss_fwd,
    _sxentropyloss_bwd)
from soket.tensor.ops.intern cimport _transpose
from soket.backend cimport Device, DeviceType, _default_device
from soket.dtype cimport DType, _default_datatype
from soket.nn.functional cimport batch_norm, layer_norm
from libc.stdlib cimport malloc, realloc
from libc.string cimport memcmp
from collections import OrderedDict


cdef class Identity(Module):
    ''' Identity module performing identity operation. '''

    def forward(Tensor X) -> Tensor:
        return X

    ## DUNDER METHODS ##

    def __str__(self) -> str:
        return 'soket.nn.Identity()'

    ## DUNDER METHODS END ##


cdef class Linear(Module):
    '''
    Constructs a linear layer applying linear transformation
    to incoming data.

      Y = X @ W + b
    '''

    # Keep track of features
    cdef object _feature_in
    cdef object _feature_out

    def __init__(
        self,
        feature_in: int, feature_out: int, bias: bool = True,
        Device device=None, DType dtype=None
    ):
        super().__init__()

        # Builtin module
        self._builtin = True

        # Store features for representation
        self._feature_in = feature_in
        self._feature_out = feature_out

        # Defaults
        device = _default_device() if device is None else device
        dtype = _default_datatype if dtype is None else dtype

        # Allocate memory for parameters
        self._nstorage = 2
        self._storage = <PyObject **> malloc(
            self._nstorage * sizeof(PyObject *)
        )
        if self._storage == NULL:
            raise MemoryError('Cannot allocate memory for linear parameters!')

        # Store weight
        ## self._storage[0] -> Weight
        cdef Tensor ten = _zeros(
            (feature_in, feature_out),
            device, dtype, True
        )
        Py_XINCREF(<PyObject *> ten)
        self._storage[0] = <PyObject *> ten

        # Store bias if needed
        ## self._storage[1] -> Bias
        ten = _zeros(
            (feature_out,),
            device, dtype, True
        ) if bias else None
        Py_XINCREF(<PyObject *> ten)
        self._storage[1] = <PyObject *> ten

    ## PROPERTIES ##

    @property
    def weight(self) -> Tensor:
        ''' Return weigth of linear transformation layer. '''

        return <Tensor> self._storage[0]

    @property
    def bias(self) -> Tensor:
        ''' Return bias of linear transformation layer. '''

        return <Tensor> self._storage[1]

    ## PROPERTIES END ##

    cdef Tensor _fast_forward(self, Tensor X, object y):
        cdef Tensor Y = X._matmul(<Tensor> self._storage[0])  # weight

        # Add bias if available
        if self._storage[1] != NULL:
            Y = Y._add(<Tensor> self._storage[1])

        return Y

    ## DUNDER METHODS ##

    def __str__(self) -> str:
        ''' Stringify the Linear module. '''

        cdef Tensor weight = self.weight

        cdef str brand = 'soket.nn.Linear('
        cdef str f_in = f'feature_in={self._feature_in}'
        cdef str f_out = f'feature_out={self._feature_out}'
        cdef str bias = f'bias={self.bias is not None}'
        cdef str dtype = f'dtype={weight._dtype._str()}'

        # Craft final string
        cdef str res = f'{brand}{f_in}, {f_out}, {bias}, {dtype}'

        if weight._device._dev_idx == DeviceType.GPU:
            res += ', device=GPU:' + str(weight._device._id)

        return res + ')'

    ## DUNDER METHODS END ##


## STRUCTURAL MODULES ##

cdef class Sequential(Module):
    ''' A sequential container executing stored modules sequentially. '''

    # Storage memory size
    cdef int _storage_size

    # Ordered dictionary details for representation
    cdef list _odict_keys
    cdef int _odict_keys_size

    def __init__(self, *modules: tuple[Module] | OrderedDict[str, Module]):
        super().__init__()

        # Builtin module
        self._builtin = True

        cdef Py_ssize_t len = PyTuple_GET_SIZE(modules)
        cdef object item

        if len > 0:
            item = <object> PyTuple_GET_ITEM(modules, 0)
            if PyObject_IsInstance(item, OrderedDict):
                self._odict_keys = list(item.keys())
                self._odict_keys_size = PyList_GET_SIZE(self._odict_keys)

                modules = tuple(item.values())
                len = PyTuple_GET_SIZE(modules)
            elif Py_TYPE(item) is <PyTypeObject *> tuple:
                modules = item
                len = PyTuple_GET_SIZE(modules)

        # Allocate storage
        self._storage = <PyObject **> malloc(
            2 * len * sizeof(PyObject *)
        )
        if self._storage == NULL:
            raise MemoryError('Failed to allocate storage to store Modules!')

        # It's safe to allocate twice more memory as model layer count
        # generally is not huge and some layers might get pushed later.
        self._storage_size = 2 * len
        self._nstorage = len

        for i in range(len):
            item = <object> PyTuple_GET_ITEM(modules, i)

            if not PyObject_IsInstance(item, Module):
                raise ValueError(f'{type(item)} is not a Module subclass!')

            Py_XINCREF(<PyObject *> item)
            self._storage[i] = <PyObject *> item

    ## METHODS ##

    cdef Tensor _fast_forward(self, Tensor X, object y):
        ''' Execute all modules in a sequential manner. '''

        # Python forward declaration
        cdef Module module

        cdef Tensor Y = X
        for i in range(self._nstorage):
            module = <Module> self._storage[i]
            if module._builtin:
                Y = module._fast_forward(Y, None)
            else:
                Y = <Tensor> module.forward(Y)

        return Y

    def append(self, Module module) -> Sequential:
        ''' Append a module at the end. '''

        # Extend memory
        if self._nstorage + 1 >= self._storage_size:
            self._storage_size *= 2
            self._storage = <PyObject **> realloc(
                self._storage, self._storage_size * sizeof(PyObject *)
            )

        Py_XINCREF(<PyObject *> module)
        self._storage[self._nstorage] = <PyObject *> module
        self._nstorage += 1

    ## METHODS END ##

    ## DUNDER METHODS ##

    def __str__(self) -> str:
        ''' Stringify sequential module. '''

        cdef object lines
        cdef str res = 'soket.nn.Sequential('
        for i in range(self._nstorage):
            # Add new line if there is atleast one module
            if i == 0:
                res += '\n'

            lines = (<Module> self._storage[i]).__str__().splitlines()
            lines = '\n'.join(lines[:1] + ['  ' + x for x in lines[1:]])

            res += '  ('
            if i < self._odict_keys_size:
                res += self._odict_keys[i]
            else:
                res += str(i)
            res += '): ' + lines + '\n'

        return res + ')'

    ## DUNDER METHODS END ##


cdef class Residual(Module):
    ''' Residual block for Residual Networks. '''
    ''' Paper: https://arxiv.org/abs/1512.03385 '''

    # Store the layers
    cdef Module _layers

    def __init__(self, layers: Module):
        super().__init__()

        # Builtin module
        self._builtin = True
        self._layers = layers

    ## CDEF METHODS ##

    cdef Tensor _fast_forward(self, Tensor X, object y):
        return X._add(self._layers(X))

    ## CDEF METHODS END ##

    ## DUNDER METHODS ##

    def __str__(self) -> str:
        ''' Stringified representation of Residual module. '''

        return f'soket.nn.Residual(layers={self._layers})'

    ## DUNDER METHODS END ##

## STRUCTURAL MODULES END ##


## ACTIVATION MODULES ##

cdef class ReLU(Module):
    ''' ReLU activation layer. '''

    def __init__(self):
        super().__init__()

        # Builtin module
        self._builtin = True

    ## CDEF METHODS ##

    cdef Tensor _fast_forward(self, Tensor X, object y):
        return Tensor._make_from_op(
            Op(_relu_fwd, _relu_bwd, OpType.RELU),
            TensorTriad(<PyObject *> X, NULL, NULL),
            X._device,
            X._dtype,
            X._shape, X._nshape,
            True,
            NULL, 0
        )

    ## CDEF METHODS END ##

    ## DUNDER METHODS ##

    def __str__(self) -> str:
        ''' Stringification of ReLU activation module. '''

        return 'soket.nn.ReLU()'

    ## DUNDER METHODS END ##

## ACTIVATION MODULES END ##


## LOSS MODULES ##

cdef class SoftmaxCrossEntropyLoss(Module):
    ''' Criterion to compute softmax cross-entropy loss. '''

    # LogSumExp reduction axes
    cdef tuple _axes_zero
    cdef tuple _axes_one

    # Preferred batch reduction
    cdef str _reduction
    cdef bint _batch_reduction

    def __init__(self, reduction: str = 'mean'):
        super().__init__()

        # Builtin module
        self._builtin = True

        # Reduction axes
        self._axes_zero = (0,)
        self._axes_one = (1,)

        if reduction != 'sum' and reduction != 'mean' and reduction != 'none':
            raise ValueError(f'Invalid reduction type - {reduction}')

        self._reduction = reduction

        # Is batch dimension getting reduced?
        if reduction == 'sum' or reduction == 'mean':
            self._batch_reduction = True

    ## CDEF METHODS ##

    cdef Tensor _fast_forward(self, Tensor X, object y):
        # Check for valid input
        if X is None or y is None:
            raise ValueError('Expected tensors as inputs, got None instead')

        cdef Tensor targets = <Tensor> y
        if X._device._ne(targets._device):
            raise RuntimeError('Incompatible device of classes and targets!')

        if X._nshape == 0:
            raise ValueError('Expected the classes tensor to be atleast 1D!')

        # Check for shape compatibility.
        # w/o batch
        if X._nshape == 1:
            if targets._nshape >= 1:
                raise ValueError(
                    'Incompatible targets tensor, expected shape to be ()'
                )
        # with batch
        elif X._nshape >= 2:
            if (targets._nshape + 1 != X._nshape or
            X._shape[0] != targets._shape[0] or

            # test channels
            memcmp(X._shape + 2, targets._shape + 1, (X._nshape - 2) *
            sizeof(int))):
                raise ValueError(
                    f'Incompatible targets tensor shape - '
                    f'{X.shape} and {targets.shape}'
                )

        # LogSumExp reduction axes
        cdef tuple axes = self._axes_one if X._nshape >= 2 else self._axes_zero

        # LogSumExp reduce axis
        cdef int r_axis = 0 + X._nshape >= 2

        # One hot encoded target
        cdef int num_classes = X._shape[r_axis]
        cdef object one_hot = _one_hot(
            targets,
            num_classes,
            X._device,
            X._dtype,
            False
        )._compute_data()

        # If dealing with extra channels
        cdef tuple permute_axis
        cdef Py_ssize_t i
        cdef int idx
        if targets._nshape > 2:
            # TODO: Performance hit is pretty bad. FIX ME.
            permute_axis = PyTuple_New(X._nshape)
            PyTuple_SET_ITEM(permute_axis, 0, 0)

            # One hot encoded dimension should be right beside the batch
            PyTuple_SET_ITEM(permute_axis, 1, X._nshape - 1)

            idx = 2
            for i in range(1, X._nshape - 1):
                PyTuple_SET_ITEM(permute_axis, idx, i)
                idx += 1

            # Permute shape
            one_hot = _transpose(X._device._dev_idx)(
                one_hot, permute_axis
            )

        # Calculate new tensor shape.
        cdef int *shape
        cdef int nshape

        if X._nshape == 1:
            # scalar
            shape = <int *> malloc(0)
            nshape = 0
        else:
            nshape = X._nshape - 1 - self._batch_reduction
            shape = <int *> malloc(nshape * sizeof(int))
            if shape == NULL:
                raise MemoryError(
                    'Failed to allocate shape for cross-entropy loss!'
                )

            idx = 0
            for i in range(self._batch_reduction, X._nshape):
                if i == r_axis:
                    continue

                shape[idx] = X._shape[i]
                idx += 1

        cdef (PyObject *)[4] value_cache = [
            <PyObject *> axes,
            <PyObject *> <object> False,
            <PyObject *> one_hot,
            <PyObject *> self._reduction,
        ]

        return Tensor._make_from_op(
            Op(
                _sxentropyloss_fwd,
                _sxentropyloss_bwd,
                OpType.SXENTROPYLOSS
            ),
            TensorTriad(<PyObject *> X, NULL, NULL),
            X._device,
            X._dtype,
            shape, nshape,
            False,
            value_cache, 4
        )

    ## CDEF METHODS END ##

    ## DUNDER METHDOS ##

    def __str__(self):
        ''' String representation of cross-entropy loss module. '''

        return (f"soket.nn.SoftmaxCrossEntropyLoss"
            f"(reduction='{self._reduction}')")

    ## DUNDER METHODS END ##

## LOSS MODULES END ##


## NORMALIZATION MODULES ##

cdef class _BatchNormBase(Module):
    ''' Batch normalization base class. '''

    # Hyper parameters
    cdef object _eps
    cdef object _momentum

    # Running mean and variance
    cdef Tensor _running_mean
    cdef Tensor _running_var

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        Device device=None,
        DType dtype=None
    ):
        super().__init__()

        # Builtin module
        self._builtin = True

        # Set hyper parameters
        self._eps = eps
        self._momentum = momentum

        # Defaults
        device = _default_device() if device is None else device
        dtype = _default_datatype if dtype is None else dtype

        # Allocate memory for parameters
        self._nstorage = 2
        self._storage = <PyObject **> malloc(
            self._nstorage * sizeof(PyObject *)
        )
        if self._storage == NULL:
            raise MemoryError('Cannot allocate memory for linear parameters!')

        # Initialize parameters
        # Store gamma (also could be interpreted as weight)
        ## self._storage[0] -> gamma
        cdef Tensor ten = _ones((num_features,), device, dtype, affine)
        Py_XINCREF(<PyObject *> ten)
        self._storage[0] = <PyObject *> ten

        # Store beta (also could be interpreted as bias)
        ## self._storage[1] -> beta
        ten = _zeros((num_features,), device, dtype, affine)
        Py_XINCREF(<PyObject *> ten)
        self._storage[1] = <PyObject *> ten

        # Running mean and variance
        if track_running_stats is True:
            self._running_mean = Tensor(0.0, device, dtype)
            self._running_var = Tensor(1.0, device, dtype)
        else:
            self._running_mean = None
            self._running_var = None

    ## CDEF METHODS ##

    cdef Tensor _fast_forward(self, Tensor X, object y):
        # Check dimension compatibility
        self._check_input_dim(X)

        return batch_norm(
            X,
            self._running_mean, self._running_var,
            <Tensor> self._storage[0], <Tensor> self._storage[1], # gamma & beta
            self._training, self._momentum, self._eps
        )

    cdef void _check_input_dim(self, Tensor X):
        raise NotImplementedError

    ## CDEF METHODS END ##

    ## DUNDER METHODS ##

    def __str__(self) -> str:
        ''' Stringify the BatchNorm  module. '''

        # For parameter datatypes
        cdef Tensor gamma = <Tensor> self._storage[0]

        cdef str brand = f'soket.nn.{self.__class__.__name__}('
        cdef str eps = f'eps={self._eps}'
        cdef str momentum = f'momentum={self._momentum}'
        cdef str affine = f'affine={gamma._requires_grad}'
        cdef str trs = f'track_running_stats={self._running_mean is not None}'
        cdef str dtype = f'dtype={gamma._dtype._str()}'

        # Craft final string
        cdef str res = (f'{brand}{self._num_features}, {eps}, {momentum}, '
            f'{affine}, {trs}, {dtype}')

        if gamma._device._dev_idx == DeviceType.GPU:
            res += ', device=GPU:' + str(gamma._device._id)

        return res + ')'

    ## DUNDER METHODS END ##


cdef class BatchNorm1d(_BatchNormBase):
    ''' Apply batch normalization over 2D and 3D inputs. '''

    ## CDEF METHODS ##

    cdef void _check_input_dim(self, Tensor X):
        if X._nshape != 2 and X._nshape != 3:
            raise ValueError('Expected 2D or 3D input tensor!')


cdef class BatchNorm2d(_BatchNormBase):
    ''' Apply batch normalization over a 4D input. '''

    ## CDEF METHODS ##

    cdef void _check_input_dim(self, Tensor X):
        if X._nshape != 4:
            raise ValueError('Expected a 4D tensor!')


cdef class BatchNorm3d(_BatchNormBase):
    ''' Apply batch normalization over a 5D input. '''

    ## CDEF METHODS ##

    cdef void _check_input_dim(self, Tensor X):
        if X._nshape != 5:
            raise ValueError('Expected a 5D input tensor!')

    ## CDEF METHODS END ##


cdef class LayerNorm(Module):
    ''' Apply Layer Normalization over 2D inputs or greater. '''

    # Store for representation
    cdef tuple _normalized_shape

    # For stability
    cdef object _eps

    def __init__(
        self,
        normalized_shape: list[int] | int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        Device device=None,
        DType dtype=None
    ):
        super().__init__()

        # Builtin module
        self._builtin = True

        normalized_shape = ((normalized_shape,) if Py_TYPE(normalized_shape) is
            <PyTypeObject *> int else tuple(normalized_shape))
        self._normalized_shape = normalized_shape
        self._eps = eps

        # Defaults
        device = _default_device() if device is None else device
        dtype = _default_datatype if dtype is None else dtype

        # Allocate memory for parameters
        self._nstorage = 2
        self._storage = <PyObject **> malloc(
            self._nstorage * sizeof(PyObject *)
        )
        if self._storage == NULL:
            raise MemoryError('Cannot allocate memory for linear parameters!')

        # Initialize parameters
        cdef Tensor ten
        if elementwise_affine is True:
            ## self._storage[0] -> weight or gamma
            ten = _ones(normalized_shape, device, dtype, True)
            Py_XINCREF(<PyObject *> ten)
            self._storage[0] = <PyObject *> ten

            ## self._storage[1] -> bias or beta
            ten = (None if bias is False else
                _zeros(normalized_shape, device, dtype, True))
            Py_XINCREF(<PyObject *> ten)
            self._storage[1] = <PyObject *> ten

    ## CDEF METHODS ##

    cdef Tensor _fast_forward(self, Tensor X, object y):
        # Call functional layernorm
        return layer_norm(
            X,
            # Weight and bias
            <Tensor> self._storage[0], <Tensor> self._storage[1],
            self._eps
        )

    ## CDEF METHODS END ##

    ## DUNDER METHODS ##

    def __str__(self) -> str:
        ''' Stringify the LayerNorm module. '''

        # For parameter datatypes
        cdef Tensor gamma = <Tensor> self._storage[0]

        cdef str brand = f'soket.nn.LayerNorm('
        cdef str eps = f'eps={self._eps}'
        cdef str affine = f'elementwise_affine={gamma._requires_grad}'
        cdef str dtype = f'dtype={gamma._dtype._str()}'

        # Craft final string
        cdef str res = (f'{brand}{self._normalized_shape}, {eps}, {affine}, '
            f'{dtype}')

        if gamma._device._dev_idx == DeviceType.GPU:
            res += ', device=GPU:' + str(gamma._device._id)

        return res + ')'

    ## DUNDER METHODS END ##

## NORMALIZATION MODULES END ##


## REGULARIZATION MODULES ##

cdef class Dropout(Module):
    ''' Dropout regularization module. '''

    # Dropout probability
    cdef object _keep_rate
    cdef object _r_keep_rate  # reciprocal of keeprate

    def __init__(self, p: float = 0.5):
        super().__init__()

        # Builtin module
        self._builtin = True
        self._keep_rate = 1.0 - p
        self._r_keep_rate = 1.0 / (1.0 - p)

    ## CDEF METHODS END ##

    cdef Tensor _fast_forward(self, Tensor X, object y):
        # Dropout is only applied during training.
        cdef Tensor binomial_mask
        if self._training:
            # Mask with keep rates.
            binomial_mask = _randb(
                X._compute_data().shape,
                self._keep_rate,
                X._device,
                X._dtype,
                False
            )
            return X._mul(binomial_mask)._mul(self._r_keep_rate)
        else:
            return X

    ## CDEF METHODS END ##

    ## DUNDER METHODS ##

    def __str__(self) -> str:
        ''' Stringify the Droptout module. '''

        return f'soket.nn.Dropout(p={1.0 - self._keep_rate})'

    ## DUNDER METHODS END ##

## REGULARIZATION MODULES END ##
