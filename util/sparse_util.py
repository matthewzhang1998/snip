import tensorflow as tf
import numpy as np

def _sparse_linear(args, output_size, bias, bias_start=0.0, scope=None):
    if args is None or (nest.is_sequence(args) and not args):
        raise ValueError("`args` must be specified")
    if not nest.is_sequence(args):
        args = [args]  # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape().as_list() for a in args]
    for shape in shapes:
        if len(shape) != 2:
            raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
        if not shape[1]:
            raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
        else:
            total_arg_size += shape[1]
            dtype = [a.dtype for a in args][0]  # Now the computation.
    num_in_weights = 10
    with vs.variable_scope(scope or "Linear"):
        def custom_initializer(shape, dtype, partition_info):
            return (tf.truncated_normal(shape, stddev=0.01, dtype=dtype) + 0.5) \
                   * tf.cast(tf.sign(tf.random_normal(shape, dtype=dtype)), dtype=dtype)
            # return tf.cast(tf.sign(tf.random_normal(shape, dtype=dtype)), dtype=dtype) * 0.1

        matrix = vs.get_variable(
            "Matrix", [output_size * num_in_weights], dtype=dtype, initializer=custom_initializer)  ## get sparse masks
        if not sparse_indices_dict.get(matrix):
            indices = np.concatenate([np.arange(output_size).reshape(output_size, 1, 1) \
              * np.int64(np.ones((output_size, num_in_weights, 1))),
              np.random.randint(0, total_arg_size,
                                [output_size, num_in_weights], ).reshape(output_size,
                                                                         num_in_weights, 1),
              ], 2).reshape(output_size * num_in_weights, 2)
            indices = map(tuple, indices)
            ## init sparse tensor
            sparse_matrix = tf.SparseTensor(indices=indices, values=matrix,
                                            dense_shape=[output_size, total_arg_size])
            sparse_indices_dict[matrix] = sparse_matrix
        else:
            # print("reuse indices")
            sparse_matrix = sparse_indices_dict[matrix]
        if len(args) == 1:
            res = tf.sparse_tensor_dense_matmul(sparse_matrix, args[0], adjoint_b=True)
        else:
            res = tf.sparse_tensor_dense_matmul(sparse_matrix, array_ops.concat(args, 1, ), adjoint_b=True)
        res = tf.transpose(res)
        ##    if not bias:
        return res
    bias_term = vs.get_variable(
        "Bias", [output_size],
        dtype=dtype,
        initializer=init_ops.constant_initializer(
            bias_start, dtype=dtype))
    return tf.nn.bias_add(res, bias_term)

class SparseDummyLSTM(object):
    def __init__(self, num_units, input_depth, seed=None,
                 init_data=None, num_unitwise=None,
                 forget_bias=None, activation=None, ):
        self._num_units = num_units
        self._forget_bias = forget_bias
        if activation:
            self._activation = get_activation(activation)
        else:
            self._activation = tf.tanh

        initializer = init([input_depth + self._num_units, 4 * self._num_units], 'kernel',
                           init_data['w_init_method'], tf.float32, init_data['w_init_para'],
                           seed=seed, trainable=True
                           )

        self._input_size = input_depth
        self._num_unitwise = num_unitwise if num_unitwise is not None else 1

        if self.dtype is None:
            initializer = tf.zeros_initializer
        else:
            initializer = tf.zeros_initializer(dtype=self.dtype)
        self._bias = tf.get_variable(name='bias',
            shape=[4 * self._num_units],
            initializer=initializer)

        self._dummy_kernel = tf.placeholder(
            shape=[input_depth + self._num_units, 4 * self._num_unitwise], dtype=tf.float32
        )
        self._dummy_bias = tf.placeholder(
            shape=[4 * self._num_unitwise], dtype=tf.float32
        )

        self._combined = self._kernel * self._mask

        if num_proj:
            self._state_size = (
                LSTMStateTuple(num_units, num_proj)
                if state_is_tuple else num_units + num_proj)
            self._output_size = num_proj
        else:
            self._state_size = (
                LSTMStateTuple(num_units, num_units)
                if state_is_tuple else 2 * num_units)
            self._output_size = num_units

    def __call__(self, inputs, state):
        num_proj = self._num_units

        c_prev = tf.slice(state, [0, 0], [-1, self._num_units])
        m_prev = tf.slice(state, [0, self._num_units], [-1, num_proj])

        input_size = inputs.get_shape().with_rank(2)[1]
        if input_size.value is None:
            raise ValueError("Could not infer input size from inputs.get_shape()[-1]")

        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        i = tf.matmul(
            tf.concat([inputs, m_prev], 1),
            self._dummy_kernel[:, :self._num_unitwise]) \
            + self._dummy_bias[:self._num_unitwise]

        j = tf.matmul(
            tf.concat([inputs, m_prev], 1),
            self._dummy_kernel[:, self._num_unitwise:2 * self._num_unitwise]) \
            + self._dummy_bias[self._num_unitwise:2 * self._num_unitwise]

        f = tf.matmul(
            tf.concat([inputs, m_prev], 1),
            self._dummy_kernel[:, 2 * self._num_unitwise:3 * self._num_unitwise]) \
            + self._dummy_bias[2 * self._num_unitwise:3 * self._num_unitwise]

        o = tf.matmul(
            tf.concat([inputs, m_prev], 1),
            self._dummy_kernel[:, 3 * self._num_unitwise:]) \
            + self._dummy_bias[3 * self._num_unitwise:]

        # Diagonal connections
        batch_size = tf.shape(inputs)[0]
        random_shape = tf.stack([batch_size, tf.constant(self._num_units - self._num_unitwise)])

        random_f = tf.random.normal(random_shape, 0, 0.3)
        random_i = tf.random.normal(random_shape, 0, 0.3)
        random_j = tf.random.normal(random_shape, 0, 0.3)
        random_o = tf.random.normal(random_shape, 0, 0.3)

        i = tf.concat([i, random_i], axis=-1)
        j = tf.concat([j, random_j], axis=-1)
        o = tf.concat([o, random_o], axis=-1)
        f = tf.concat([f, random_f], axis=-1)

        c = (tf.sigmoid(f + self._forget_bias) * c_prev + tf.sigmoid(i) *
             self._activation(j))

        m = tf.sigmoid(o) * self._activation(c)

        new_state = tf.concat([c, m], 1)
        return m, new_state

class SparseDummyGRU(object):
    def __init__(self, params):

    def __call__(self, input):


def generate_sparse_parameters():
