## DNN MODULE PROTOTYPES ##

from cpython.ref cimport Py_XINCREF
from cpython.object cimport Py_TYPE, PyTypeObject, PyObject, PyObject_IsInstance
from cpython.tuple cimport (PyTuple_GET_SIZE, PyTuple_GET_ITEM, PyTuple_New,
    PyTuple_SET_ITEM)
from cpython.list cimport PyList_GET_SIZE
from soket.nn.module cimport Module
from soket.tensor cimport Tensor, Op, OpType, TensorTriad
from soket.tensor.creation cimport _randb, _ones, _zeros
from soket.tensor.detached cimport tanh
from soket.tensor.ops cimport _relu_fwd, _relu_bwd
from soket.tensor.ops.intern cimport _transpose
from soket.backend cimport Device, DeviceType, _default_device
from soket.dtype cimport DType, _default_datatype
from soket.nn.functional cimport *
from libc.stdlib cimport malloc, realloc
from libc.string cimport memcmp
from collections import OrderedDict
from soket.tensor import rand, randn


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

        # Initialize and store parameters
        cdef object k = 1 / feature_in

        # weight
        ## self._storage[0] -> Weight
        cdef Tensor ten = rand(
            (feature_in, feature_out),
            low=-k, high=k,
            device=device, dtype=dtype,
            requires_grad=True
        )
        Py_XINCREF(<PyObject *> ten)
        self._storage[0] = <PyObject *> ten

        # bias (if needed)
        ## self._storage[1] -> Bias
        ten = rand(
            (feature_out,),
            low=-k, high=k,
            device=device, dtype=dtype,
            requires_grad=True
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
        cdef Tensor bias = <Tensor> self._storage[1]
        if bias != None:
            Y = Y._add(bias)

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


cdef class Tanh(Module):
    ''' Tanh activation layer. '''

    def __init__(self):
        super().__init__()

        # Builtin module
        self._builtin = True

    ## CDEF METHODS ##

    cdef Tensor _fast_forward(self, Tensor X, object y):
        return tanh(X)

    ## CDEF METHODS END ##

    ## DUNDER METHODS ##

    def __str__(self) -> str:
        ''' Stringification of Tanh activation module. '''

        return 'soket.nn.Tanh()'

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

        if reduction != 'sum' and reduction != 'mean' and reduction != 'none':
            raise ValueError(f'Invalid reduction type - {reduction}')

        self._reduction = reduction

    ## CDEF METHODS ##

    cdef Tensor _fast_forward(self, Tensor X, object y):
        return softmax_cross_entropy(X, <Tensor> y, self._reduction)

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
    cdef object _num_features
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

        # Set hyperparameters
        self._num_features = num_features
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


## EMBEDDING ##

cdef class Embedding(Module):
    '''
    A look-up table storing the embedding vectors of specific dictionary size.
    '''

    # Hyperparameters
    cdef object _num_embeddings
    cdef object _embedding_dim

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        Device device=None,
        DType dtype=None
    ):
        super().__init__()

        # Builtin module
        self._builtin = True
        self._num_embeddings = num_embeddings
        self._embedding_dim = embedding_dim

        # Allocate memory for weight
        self._nstorage = 1
        self._storage = <PyObject **> malloc(
            self._nstorage * sizeof(PyObject *)
        )
        if self._storage == NULL:
            raise MemoryError('Cannot allocate memory to store embedding weight!')

        # self._storage[0] -> Weight
        cdef Tensor ten = randn(
            (num_embeddings, embedding_dim),
            device=device,
            dtype=dtype,
            requires_grad=True
        )
        Py_XINCREF(<PyObject *> ten)
        self._storage[0] = <PyObject *> ten

    ## PROPERTIES ##

    @property
    def weight(self) -> Tensor:
        ''' Return the embedding lookup table or the weight. '''

        return (<Tensor> self._storage[0])

    ## PROPERTIES END ##

    ## CDEF METHODS ##

    cdef Tensor _fast_forward(self, Tensor X, object y):
        return embedding(X, <Tensor> self._storage[0])

    ## CDEF METHODS END ##

    ## DUNDER METHODS ##

    def __str__(self) -> str:
        ''' Stringify the Embedding module. '''

        cdef str res = (f'soket.nn.Embedding({self._num_embeddings}, '
            f'{self._embedding_dim}')

        cdef Tensor weight = self.weight
        if weight._device._dev_idx == DeviceType.GPU:
            res += f', device=GPU:{int(weight._device._id)}'

        return res + ')'

    ## DUNDER METHODS END ##

## EMBEDDING END ##
