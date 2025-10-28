from cpython.ref cimport Py_XINCREF
from cpython.tuple cimport PyTuple_New, PyTuple_GET_ITEM, PyTuple_SET_ITEM
from cpython.long cimport PyLong_FromLong
from soket.tensor.ops.intern cimport *
from soket.tensor.creation cimport _zeros_like
from soket.dtype cimport DType
from libc.stdlib cimport malloc
from libc.string cimport memcpy
from math import log


## HELPER FUNCTIONS ##

cdef Tensor _create_tensor_node_like(Tensor node, object data, DType dtype):
    ''' Creates a given tensor alike, but with different data. '''

    return Tensor._make_const(
        data,
        node._device,
        dtype,
        node._shape, node._nshape,
        True,
        False
    )

cdef Tensor _create_matmul_tensor_node_like(
    Tensor node,
    Tensor input,
    object data,
    DType dtype
):
    '''
    Creates a matmul result tensor, with identical
    (but not equal) shape as node.
    '''

    cdef int *shape = <int *> malloc(node._nshape * sizeof(int))
    cdef int nshape = node._nshape

    # Copy node shape except last two dimension
    memcpy(shape, node._shape, (nshape - 2) * sizeof(int))

    # Set last two dimension of input
    shape[nshape - 1] = input._shape[input._nshape - 1]
    shape[nshape - 2] = input._shape[input._nshape - 2]

    return Tensor._make_const(
        data,
        node._device,
        dtype,
        shape, nshape,
        False,
        False
    )


## HELPER FUNCTIONS END ##


cdef TensorTriad _elemwise_add_bwd(
    Tensor node,
    Tensor adj,
    Tensor x, Tensor y, Tensor Z
):
    ''' Backward pass for element-wise add operation. '''

    cdef TensorTriad output = TensorTriad(NULL, NULL, NULL)
    cdef Tensor grad_x, grad_y

    # Input `x` gradient
    if x._requires_grad:
        ## adj (copy)

        grad_x = Tensor(adj, None, x._dtype)
        Py_XINCREF(<PyObject *> grad_x)
        output.x = <PyObject *> grad_x

    # Input `y` gradient
    if y._requires_grad:
        ## adj (copy)

        grad_y = Tensor(adj, None, y._dtype)
        Py_XINCREF(<PyObject *> grad_y)
        output.y = <PyObject *> grad_y

    return output


cdef TensorTriad _scalar_add_bwd(
    Tensor node,
    Tensor adj,
    Tensor x, Tensor y, Tensor Z
):
    ''' Backward pass for scalar add operation. '''

    cdef TensorTriad output = TensorTriad(NULL, NULL, NULL)
    cdef Tensor grad

    # Input `x`
    if x._requires_grad:
        ## adj (copy)

        grad = Tensor(adj, None, x._dtype)
        Py_XINCREF(<PyObject *> grad)
        output.x = <PyObject *> grad

    return output


cdef TensorTriad _negate_bwd(
    Tensor node,
    Tensor adj,
    Tensor x, Tensor y, Tensor Z
):
    ''' Backward pass for negation operation. '''

    cdef TensorTriad output = TensorTriad(NULL, NULL, NULL)
    cdef Tensor res

    # Input `x`
    if x._requires_grad:
        ## -adj

        res = _create_tensor_node_like(
            node,
            _neg(node._device._dev_idx)(adj._compute_data()),
            x._dtype
        )
        Py_XINCREF(<PyObject *> res)
        output.x = <PyObject *> res

    return output

cdef TensorTriad _elemwise_sub_bwd(
    Tensor node,
    Tensor adj,
    Tensor x, Tensor y, Tensor Z
):
    ''' Backward pass for element-wise subtract operation. '''

    cdef TensorTriad output = TensorTriad(NULL, NULL, NULL)
    cdef int didx = node._device._dev_idx
    cdef Tensor grad_x, grad_y
    cdef object grad_data

    # Input `x` gradient
    if x._requires_grad:
        ## adj (copy)

        grad_x = Tensor(adj, None, x._dtype)
        Py_XINCREF(<PyObject *> grad_x)
        output.x = <PyObject *> grad_x

    # Input `y` gradient
    if y._requires_grad:
        ## -adj

        grad_data = _neg(didx)(adj._compute_data())
        if adj._dtype._idx != y._dtype._idx:
            grad_data = _array(didx)(
                grad_data, y._dtype._str()
            )

        grad_y = _create_tensor_node_like(
            node,
            grad_data,
            y._dtype
        )
        Py_XINCREF(<PyObject *> grad_y)
        output.y = <PyObject *> grad_y

    return output


cdef TensorTriad _scalar_sub_bwd(
    Tensor node,
    Tensor adj,
    Tensor x, Tensor y, Tensor Z
):
    ''' Backward pass for scalar subtract operation '''

    cdef TensorTriad output = TensorTriad(NULL, NULL, NULL)
    cdef int didx = node._device._dev_idx
    cdef Tensor grad
    cdef object grad_data

    # Don't commute
    if <int> node._value_cache[1] == 0:
        if x._requires_grad:
            ## adj (copy)

            grad = Tensor(adj, None, x._dtype)
            Py_XINCREF(<PyObject *> grad)
            output.x = <PyObject *> grad
    elif y._requires_grad:
        ## -adj

        grad_data = _neg(didx)(adj._compute_data())
        if adj._dtype._idx != y._dtype._idx:
            grad_data = _array(didx)(
                grad_data, y._dtype._str()
            )

        grad = _create_tensor_node_like(
            node,
            grad_data,
            y._dtype
        )
        Py_XINCREF(<PyObject *> grad)
        output.y = <PyObject *> grad

    return output


cdef TensorTriad _elemwise_mul_bwd(
    Tensor node,
    Tensor adj,
    Tensor x, Tensor y, Tensor Z
):
    ''' Backward pass for element-wise multiply operation. '''

    cdef TensorTriad output = TensorTriad(NULL, NULL, NULL)
    cdef Tensor grad_x, grad_y

    # Input `x`
    if x._requires_grad:
        ## adj * y

        grad_x = _create_tensor_node_like(
            node,
            _mul(node._device._dev_idx)(
                adj._compute_data(),
                y._compute_data(),
                dtype=x._dtype._str()
            ),
            x._dtype
        )
        Py_XINCREF(<PyObject *> grad_x)
        output.x = <PyObject *> grad_x

    # Input `y`
    if y._requires_grad:
        ## adj * x

        grad_y = _create_tensor_node_like(
            node,
            _mul(node._device._dev_idx)(
                adj._compute_data(),
                x._compute_data(),
                dtype=y._dtype._str()
            ),
            y._dtype
        )
        Py_XINCREF(<PyObject *> grad_y)
        output.y = <PyObject *> grad_y

    return output

cdef TensorTriad _scalar_mul_bwd(
    Tensor node,
    Tensor adj,
    Tensor x, Tensor y, Tensor Z
):
    ''' Backward pass for scalar multiplication operation. '''

    cdef TensorTriad output = TensorTriad(NULL, NULL, NULL)
    cdef Tensor grad

    # Input `x`
    if x._requires_grad:
        ## adj * scalar

        grad = _create_tensor_node_like(
            node,
            _mul(node._device._dev_idx)(
                adj._compute_data(),
                <object> node._value_cache[0],  # Scalar
                dtype=x._dtype._str()
            ),
            x._dtype
        )
        Py_XINCREF(<PyObject *> grad)
        output.x = <PyObject *> grad

    return output


cdef TensorTriad _elemwise_div_bwd(
    Tensor node,
    Tensor adj,
    Tensor x, Tensor y, Tensor Z
):
    ''' Backward pass for element-wise division operation. '''

    cdef TensorTriad output = TensorTriad(NULL, NULL, NULL)
    cdef int didx = node._device._dev_idx
    cdef object y_data = y._compute_data()
    cdef Tensor grad_x, grad_y

    # Input `x` gradient
    if x._requires_grad:
        ## adj / y

        grad_x = _create_tensor_node_like(
            node,
            _div(didx)(
                adj._compute_data(),
                y._compute_data(),
                dtype=x._dtype._str()
            ),
            x._dtype
        )
        Py_XINCREF(<PyObject *> grad_x)
        output.x = <PyObject *> grad_x

    # Input `y` gradient
    if y._requires_grad:
        ## -adj * x / (y * y)

        grad_y = _create_tensor_node_like(
            node,
            _neg(didx)(_mul(didx)(
                adj._compute_data(),
                _div(didx)(x._compute_data(), _mul(didx)(
                    y_data, y_data
                )),
                dtype=y._dtype._str()
            )),
            y._dtype
        )
        Py_XINCREF(<PyObject *> grad_y)
        output.y = <PyObject *> grad_y

    return output


cdef TensorTriad _scalar_div_bwd(
    Tensor node,
    Tensor adj,
    Tensor x, Tensor y, Tensor Z
):
    ''' Backward pass for scalar division operation. '''

    cdef TensorTriad output = TensorTriad(NULL, NULL, NULL)
    cdef int didx = node._device._dev_idx
    cdef object reciprocal_scalar
    cdef Tensor grad

    # Don't commute
    if <int> node._value_cache[1] == 0:
        if x._requires_grad:
            ## adj * (1 / scalar)

            reciprocal_scalar = 1 / <object> node._value_cache[0]  # Scalar
            grad = _create_tensor_node_like(
                node,
                _mul(didx)(
                    adj._compute_data(),
                    reciprocal_scalar,
                    dtype=x._dtype._str()
                ),
                x._dtype
            )
            Py_XINCREF(<PyObject *> grad)
            output.x = <PyObject *> grad
    elif y._requires_grad:
        ## adj * -scalar * (y ** -2)
        grad = _create_tensor_node_like(
            node,
            _mul(didx)(
                adj._compute_data(),
                _mul(didx)(- <object> node._value_cache[1], _pow(didx)(
                    y._compute_data(), -2
                )),
                dtype=y._dtype._str()
            ),
            y._dtype
        )
        Py_XINCREF(<PyObject *> grad)
        output.y = <PyObject *> grad

    return output


cdef TensorTriad _elemwise_pow_bwd(
    Tensor node,
    Tensor adj,
    Tensor x, Tensor y, Tensor Z
):
    ''' Backward pass for element-wise raise to a power. '''

    cdef TensorTriad output = TensorTriad(NULL, NULL, NULL)
    cdef int didx = node._device._dev_idx
    cdef Tensor grad_x, grad_y
    cdef object y_data

    # Input `x` gradient
    if x._requires_grad:
        ## adj * y * (x ** (y - 1))

        y_data = y._compute_data()
        grad_x = _create_tensor_node_like(
            node,
            _mul(didx)(
                adj._compute_data(),
                _mul(didx)(
                    y_data,
                    _pow(didx)(
                        x._compute_data(),
                        _sub(didx)(y_data, 1)
                    )
                ),
                dtype=x._dtype._str()
            ),
            x._dtype
        )
        Py_XINCREF(<PyObject *> grad_x)
        output.x = <PyObject *> grad_x

    # Input `y` gradient
    if y._requires_grad:
        ## adj * node * log(x)

        grad_y = _create_tensor_node_like(
            node,
            _mul(didx)(
                adj._compute_data(),
                _mul(didx)(
                    node._compute_data(),
                    _log(didx)(x._compute_data())
                ),
                dtype=y._dtype._str()
            ),
            y._dtype
        )
        Py_XINCREF(<PyObject *> grad_y)
        output.y = <PyObject *> grad_y

    return output


cdef TensorTriad _scalar_pow_bwd(
    Tensor node,
    Tensor adj,
    Tensor x, Tensor y, Tensor Z
):
    ''' Backward pass for scalar raise to a power or vice versa. '''

    cdef TensorTriad output = TensorTriad(NULL, NULL, NULL)
    cdef int didx = node._device._dev_idx
    cdef object scalar
    cdef Tensor grad

    # Don't commute
    if <int> node._value_cache[1] == 0:
        if x._requires_grad:
            ## adj * scalar * (x ** (scalar - 1))

            scalar = <object> node._value_cache[0]  # Scalar
            grad = _create_tensor_node_like(
                node,
                _mul(didx)(
                    adj._compute_data(),
                    _mul(didx)(
                        scalar,
                        _pow(didx)(x._compute_data(), scalar - 1)
                    ),
                    dtype=x._dtype._str()
                ),
                x._dtype
            )
            Py_XINCREF(<PyObject *> grad)
            output.x = <PyObject *> grad
    elif y._requires_grad:
        ## adj * node * log(scalar)

        grad = _create_tensor_node_like(
            node,
            _mul(didx)(
                adj._compute_data(),
                _mul(didx)(
                    node._compute_data(),
                    log(<object> node._value_cache[1])
                ),
                dtype=y._dtype._str()
            ),
            y._dtype
        )
        Py_XINCREF(<PyObject *> grad)
        output.y = <PyObject *> grad

    return output


cdef TensorTriad _broadcast_to_bwd(
    Tensor node,
    Tensor adj,
    Tensor x, Tensor y, Tensor Z
):
    ''' Backward pass for broadcasting operation. '''

    cdef TensorTriad output = TensorTriad(NULL, NULL, NULL)
    cdef int didx = node._device._dev_idx
    cdef int diff
    cdef object data
    cdef Tensor grad

    cdef Py_ssize_t i
    cdef Py_ssize_t reduction_size
    cdef int reduction_idx
    cdef tuple reduction_shape

    if x._requires_grad:
        ## adj.sum((broadcasted_axes,))[.reshape(original_shape)]

        diff = node._nshape - x._nshape

        ## NOTE: This difference helps account for scalar tensors as well.
        reduction_size = diff

        for i in range(x._nshape):
            if node._shape[diff + i] != x._shape[i]:
                reduction_size += 1

        reduction_shape = PyTuple_New(reduction_size)
        reduction_idx = 0
        for i in range(node._nshape):
            if i < diff or node._shape[i] != x._shape[i - diff]:
                PyTuple_SET_ITEM(reduction_shape, reduction_idx, i)
                reduction_idx += 1

        data = _sum(didx)(
            adj._compute_data(), reduction_shape,
            None, None, True
        )

        if diff != 0:
            data = _reshape(didx)(data, x._compute_data().shape)

        # Exception, use `x` shape instead
        grad = _create_tensor_node_like(x, data, x._dtype)
        Py_XINCREF(<PyObject *> grad)
        output.x = <PyObject *> grad

    return output


cdef TensorTriad _sum_bwd(
    Tensor node,
    Tensor adj,
    Tensor x, Tensor y, Tensor Z
):
    ''' Backward pass for summation reduction. '''

    cdef TensorTriad output = TensorTriad(NULL, NULL, NULL)
    cdef int didx = node._device._dev_idx
    cdef object adj_data = adj._compute_data()
    cdef tuple x_shape = <tuple> x._compute_data().shape
    cdef Tensor grad
    cdef object data

    cdef tuple bcast_axes
    cdef int bcast_axes_idx
    cdef int node_idx

    if x._requires_grad:
        ## adj[.reshape()].broadcast_to(x_shape)

        if <object> node._value_cache[1] is True:  # keepdims
            data = _broadcast_to(didx)(adj_data, x_shape)
            if x._dtype._idx != node._dtype._idx:  # If dtype not same
                data = _array(didx)(data, x._dtype._str())

            grad = _create_tensor_node_like(x, data, x._dtype)
            Py_XINCREF(<PyObject *> grad)
            output.x = <PyObject *> grad

            return output

        bcast_axes = PyTuple_New(x._nshape)
        bcast_axes_idx = 0
        node_idx = 0
        for i in range(x._nshape):
            # Also account scalar tensor while summing over all axis
            if node._nshape > 0 and x._shape[i] == node._shape[node_idx]:
                PyTuple_SET_ITEM(bcast_axes, bcast_axes_idx, x._shape[i])
                node_idx += 1
            else:
                PyTuple_SET_ITEM(bcast_axes, bcast_axes_idx, 1)

            bcast_axes_idx += 1

        data = _broadcast_to(didx)(
            _reshape(didx)(adj_data, bcast_axes),
            x_shape
        )
        if x._dtype._idx != node._dtype._idx:  # If dtype not same
            data = _array(didx)(data, x._dtype._str())

        grad = _create_tensor_node_like(x, data, x._dtype)
        Py_XINCREF(<PyObject *> grad)
        output.x = <PyObject *> grad

    return output


cdef TensorTriad _mean_bwd(
    Tensor node,
    Tensor adj,
    Tensor x, Tensor y, Tensor Z
):
    ''' Backward pass for mean reduction. '''

    # Python scope forward declaration
    cdef object reciprocal_observations
    cdef TensorTriad output
    cdef Tensor grad

    if x._requires_grad:
        reciprocal_observations = 1 / <object> node._value_cache[2]
        output = _sum_bwd(node, adj, x, y, None)
        grad = <Tensor> output.x
        grad._data_ = _mul(node._device._dev_idx)(
            grad._compute_data(),
            reciprocal_observations,
        )

    return output

cdef TensorTriad _max_bwd(
    Tensor node,
    Tensor adj,
    Tensor x, Tensor y, Tensor Z
):
    ''' Backward pass of max operation on data. '''

    cdef TensorTriad output = TensorTriad(NULL, NULL, NULL)
    cdef int didx = node._device._dev_idx
    cdef Tensor grad
    cdef object x_data
    cdef object node_data
    cdef object adj_data
    cdef object mask
    cdef object count
    cdef tuple retain_shape
    cdef int retain_shape_idx
    cdef int node_idx

    if x._requires_grad:
        x_data = x._compute_data()
        node_data = node._compute_data()
        adj_data = adj._compute_data()

        # If keepdims == False
        if <object> node._value_cache[1] is False:
            retain_shape = x_data.shape

            retain_shape_idx = 0
            node_idx = 0
            for i in range(x._nshape):
                if node._nshape > 0 and x._shape[i] == node._shape[node_idx]:
                    node_idx += 1
                else:
                    PyTuple_SET_ITEM(retain_shape, i, 1)

            node_data = _reshape(didx)(node_data, retain_shape)
            adj_data = _reshape(didx)(adj_data, retain_shape)

        # Mask array same shape as `x` denoting where maximum value was found.
        mask = _equal(didx)(x_data, node_data)

        # If multiple max value is found over an axis
        count = _sum(didx)(
            mask,
            <tuple> node._value_cache[0],
            None, None, True
        )

        grad = _create_tensor_node_like(
            x,
            _mul(didx)(
                _div(didx)(adj_data, count),
                mask
            ),
            x._dtype
        )

        Py_XINCREF(<PyObject *> grad)
        output.x = <PyObject *> grad

    return output


cdef TensorTriad _min_bwd(
    Tensor node,
    Tensor adj,
    Tensor x, Tensor y, Tensor Z
):
    ''' Backward pass of min operation on data. '''

    return _max_bwd(node, adj, x, y, None)


cdef TensorTriad _matmul_bwd(
    Tensor node,
    Tensor adj,
    Tensor x, Tensor y, Tensor Z
):
    ''' Backward pass for matmul operation. '''

    cdef TensorTriad output = TensorTriad(NULL, NULL, NULL)
    cdef int didx = node._device._dev_idx
    cdef object adj_data = adj._compute_data()
    cdef Tensor grad_x, grad_y

    # Input `x`
    if x._requires_grad:
        grad_x = _create_matmul_tensor_node_like(
            node, x,
            _matmul(didx)(
                adj_data,
                y._compute_data().T
            ),
            node._dtype
        )
        Py_XINCREF(<PyObject *> grad_x)
        output.x = <PyObject *> grad_x

    # Input `y`
    if y._requires_grad:
        grad_y = _create_matmul_tensor_node_like(
            node, y,
            _matmul(didx)(
                x._compute_data().T,
                adj_data
            ),
            node._dtype
        )
        Py_XINCREF(<PyObject *> grad_y)
        output.y = <PyObject *> grad_y

    return output

cdef TensorTriad _reshape_bwd(
    Tensor node,
    Tensor adj,
    Tensor x, Tensor y, Tensor Z
):
    ''' Backward pass for reshape operation. '''

    cdef TensorTriad output = TensorTriad(NULL, NULL, NULL)
    cdef Tensor grad

    if x._requires_grad:
        grad = _create_tensor_node_like(
            x,
            _reshape(node._device._dev_idx)(
                adj._compute_data(),
                x._compute_data().shape
            ),
            x._dtype
        )

        Py_XINCREF(<PyObject *> grad)
        output.x = <PyObject *> grad

    return output

cdef TensorTriad _permute_bwd(
    Tensor node,
    Tensor adj,
    Tensor x, Tensor y, Tensor Z
):
    ''' Backward pass for permute operation. '''

    cdef TensorTriad output = TensorTriad(NULL, NULL, NULL)
    cdef tuple axes = <tuple> node._value_cache[0]
    cdef Tensor grad
    cdef tuple inv_axes
    cdef int axis

    if x._requires_grad:
        inv_axes = PyTuple_New(node._nshape)
        for i in range(node._nshape):
            axis = <int> <object> PyTuple_GET_ITEM(axes, i)

            # Adjust axis
            axis = axis + node._nshape if axis < 0 else axis
            PyTuple_SET_ITEM(inv_axes, axis, PyLong_FromLong(i))

        grad = _create_tensor_node_like(
            x,
            _transpose(node._device._dev_idx)(
                adj._compute_data(),
                inv_axes
            ),
            x._dtype
        )

        Py_XINCREF(<PyObject *> grad)
        output.x = <PyObject *> grad

    return output


cdef TensorTriad _transpose_bwd(
    Tensor node,
    Tensor adj,
    Tensor x, Tensor y, Tensor Z
):
    ''' Backward pass for transpose operation. '''

    cdef TensorTriad output = TensorTriad(NULL, NULL, NULL)
    cdef Tensor grad

    if x._requires_grad:
        grad = _create_tensor_node_like(
            x,
            adj._compute_data().T,
            x._dtype
        )

        Py_XINCREF(<PyObject *> grad)
        output.x = <PyObject *> grad

    return output


cdef TensorTriad _select_bwd(
    Tensor node,
    Tensor adj,
    Tensor x, Tensor y, Tensor Z
):
    ''' Backward pass for select operation. '''

    cdef TensorTriad output = TensorTriad(NULL, NULL, NULL)
    cdef Tensor grad

    if x._requires_grad:
        grad = _zeros_like(x, node._device, x._dtype, False)
        grad.__setitem__(<object> node._value_cache[0], adj)

        Py_XINCREF(<PyObject *> grad)
        output.x = <PyObject *> grad

    return output


cdef TensorTriad _relu_bwd(
    Tensor node,
    Tensor adj,
    Tensor x, Tensor y, Tensor Z
):
    ''' Backward pass for ReLU activation operation. '''

    cdef TensorTriad output = TensorTriad(NULL, NULL, NULL)
    cdef int didx = node._device._dev_idx
    cdef Tensor grad

    if x._requires_grad:
        grad = _create_tensor_node_like(
            node,
            _mul(didx)(
                _greater(didx)(x._compute_data(), 0),
                adj._compute_data(),
                dtype=node._dtype._str()
            ),
            node._dtype
        )

        Py_XINCREF(<PyObject *> grad)
        output.x = <PyObject *> grad

    return output


cdef TensorTriad _log_bwd(
    Tensor node,
    Tensor adj,
    Tensor x, Tensor y, Tensor Z
):
    ''' Backward pass for log operation. '''

    cdef TensorTriad output = TensorTriad(NULL, NULL, NULL)
    cdef Tensor grad

    if x._requires_grad:
        grad = _create_tensor_node_like(
            node,
            _div(node._device._dev_idx)(
                adj._compute_data(),
                x._compute_data()
            ),
            node._dtype
        )

        Py_XINCREF(<PyObject *> grad)
        output.x = <PyObject *> grad

    return output


cdef TensorTriad _exp_bwd(
    Tensor node,
    Tensor adj,
    Tensor x, Tensor y, Tensor Z
):
    ''' Backward pass for exponential operation. '''

    cdef TensorTriad output = TensorTriad(NULL, NULL, NULL)
    cdef Tensor grad

    if x._requires_grad:
        grad = _create_tensor_node_like(
            node,
            _mul(node._device._dev_idx)(
                adj._compute_data(),
                node._compute_data()
            ),
            node._dtype
        )

        Py_XINCREF(<PyObject *> grad)
        output.x = <PyObject *> grad

    return output


cdef TensorTriad _logsumexp_bwd(
    Tensor node,
    Tensor adj,
    Tensor x, Tensor y, Tensor Z
):
    ''' Backward pass for log-sum-exp operation. '''

    cdef TensorTriad output = TensorTriad(NULL, NULL, NULL)
    cdef int didx = node._device._dev_idx
    cdef Tensor grad
    cdef object axes = <object> node._value_cache[0]
    cdef object x_data = x._compute_data()
    cdef object max, exp_x, sum_exp_x

    if x._requires_grad:
        output = _sum_bwd(node, adj, x, y, None)
        grad = <Tensor> output.x

        max = _max(didx)(x_data, axes, None, True)
        exp_x = _exp(didx)(_sub(didx)(x_data, max))
        sum_exp_x = _sum(didx)(exp_x, axes, None, None, True)

        grad._data_ = _div(didx)(
            _mul(didx)(grad._data_, exp_x),
            sum_exp_x
        )

    return output


cdef TensorTriad _sxentropyloss_bwd(
    Tensor node,
    Tensor adj,
    Tensor x, Tensor y, Tensor Z
):
    ''' Backward pass for softmax cross entropy loss. '''

    cdef TensorTriad output = TensorTriad(NULL, NULL, NULL)
    cdef int didx = node._device._dev_idx
    cdef reduction = <object> node._value_cache[3]
    cdef object x_data = x._compute_data()
    cdef object adj_data = adj._compute_data()
    cdef Tensor grad
    cdef object max, exp_x, sum_exp_x
    cdef object batch_grad
    cdef double reciprocal_batch_size
    cdef tuple new_shape

    if x._requires_grad:
        ## adj * (softmax(x) - one_hot)

        max = _max(didx)(x_data, <object> node._value_cache[0], None, True)
        exp_x = _exp(didx)(_sub(didx)(x_data, max))
        sum_exp_x = _sum(didx)(
            exp_x,
            <object> node._value_cache[0],
            None, None, True
        )

        batch_grad = _sub(didx)(
            _div(didx)(exp_x, sum_exp_x),
            <object> node._value_cache[2],
            dtype=x._dtype._str()
        )

        if reduction == 'mean' and x._nshape >= 2:
            reciprocal_batch_size = 1 / <double> x._shape[0]
            batch_grad = _mul(didx)(batch_grad, reciprocal_batch_size)
        elif reduction == 'none' and x._nshape >= 2:
            new_shape = PyTuple_New(x._nshape)

            # Set batch dimension
            PyTuple_SET_ITEM(new_shape, 0, x._shape[0])

            # Broadcastable class dimension
            PyTuple_SET_ITEM(new_shape, 1, 1)

            # Channel dimensions stay the same
            for i in range(2, x._nshape):
                PyTuple_SET_ITEM(new_shape, i, x._shape[i])

            # Reshape adjoint
            adj_data = _reshape(didx)(adj_data, new_shape)

        grad = _create_tensor_node_like(
            x,
            _mul(didx)(adj_data, batch_grad),
            x._dtype
        )

        Py_XINCREF(<PyObject *> grad)
        output.x = <PyObject *> grad

    return output


cdef TensorTriad _bnorm_bwd(
    Tensor node,
    Tensor adj,
    Tensor x, Tensor y, Tensor Z
):
    ''' Backward pass for Batch Normalization operation. '''

    cdef TensorTriad output = TensorTriad(NULL, NULL, NULL)
    cdef int didx = node._device._dev_idx
    cdef object adj_data = adj._compute_data()
    cdef object reduce_axes = <object> node._value_cache[0]
    cdef object observ = <object> node._value_cache[1]
    cdef Tensor running_mean = <Tensor> node._value_cache[2]
    cdef object momentum = <object> node._value_cache[5]
    cdef object xshift = <object> node._value_cache[7]
    cdef object rvar = <object> node._value_cache[8]
    cdef object norm = <object> node._value_cache[9]
    cdef bint layernorm = running_mean is None and momentum is None

    # Intermediate values
    cdef object dxnorm
    cdef Tensor grad
    cdef object robserv

    # Gradient reduction for gamma and beta is different for batchnorm and
    # layernorm.
    cdef tuple xy_reduce_axes
    if layernorm:
        xy_reduce_axes = (0,)
    else:
        xy_reduce_axes = reduce_axes

    # gamma
    if x is not None and x._requires_grad:
        grad = _create_tensor_node_like(
            x,
            _sum(didx)(
                _mul(didx)(norm, adj_data),
                xy_reduce_axes, None, None, False
            ),
            x._dtype
        )
        Py_XINCREF(<PyObject *> grad)
        output.x = <PyObject *> grad

    # beta
    if y is not None and y._requires_grad:
        grad = _create_tensor_node_like(
            y,
            _sum(didx)(adj_data, xy_reduce_axes, None, None, False),
            y._dtype
        )
        Py_XINCREF(<PyObject *> grad)
        output.y = <PyObject *> grad

    # X
    if Z._requires_grad:
        robserv = 1.0 / observ

        # Partial adjoint w.r.t. X
        if layernorm:  # layernorm
            dxnorm = (_mul(didx)(adj_data, x._compute_data())
                if x is not None else adj_data)
        else:  # batchnorm
            dxnorm = (_mul(didx)(
                adj_data,
                _reshape(didx)(x._compute_data(), rvar.shape)
            ) if x is not None else adj_data)

        # Partial adjoint w.r.t variance
        dvar = _sum(didx)(
            _mul(didx)(
                _mul(didx)(dxnorm, xshift),
                _mul(didx)(-0.5, _mul(didx)(_mul(didx)(rvar, rvar), rvar))
            ),
            reduce_axes, None, None, True
        )

        # Partial adjoint w.r.t mean
        dmean = _add(didx)(
            _sum(didx)(
                _mul(didx)(dxnorm, _mul(didx)(-1.0, rvar)),
                reduce_axes, None, None, True
            ),
            _mul(didx)(dvar, _mul(didx)(
                robserv, _sum(didx)(
                    _mul(didx)(-2.0, xshift),
                    reduce_axes, None, None, True
                )
            ))
        )

        # Gradient w.r.t input Z
        grad = _create_tensor_node_like(
            Z,
            _add(didx)(_mul(didx)(robserv, dmean), _add(didx)(
                _mul(didx)(dxnorm, rvar),
                _mul(didx)(
                    _mul(didx)(dvar, (2.0 * robserv)),
                    xshift
                )
            )),
            Z._dtype
        )
        Py_XINCREF(<PyObject *> grad)
        output.Z = <PyObject *> grad

    return output
