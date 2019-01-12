import tensorflow as tf

from util.network_util import *

class DenseFullyConnected(object):

    def __init__(self, input_depth, hidden_size, scope,
                 activation_type, normalizer_type, init_matrix,
                 train=True):

        self._scope = scope
        self.input_depth = input_depth
        self.hidden_size = hidden_size

        print(init_matrix)

        with tf.variable_scope(self._scope):
            self.weight = tf.Variable(init_matrix, dtype=tf.float32)
            self._b = tf.Variable(tf.zeros(shape=[hidden_size], dtype=tf.float32))
        self._train = train

        self._activation_type = activation_type
        self._normalizer_type = normalizer_type
        self.initialize_op = tf.initialize_variables([self._b, self.weight])

    def __call__(self, input_vec):
        output_shape = tf.shape(input_vec)
        flat_input = tf.reshape(input_vec,
            tf.concat([[-1], [output_shape[-1]]], axis=0)
        )

        with tf.variable_scope(self._scope):
            res = tf.matmul(flat_input, self.weight) + self._b

            if self._activation_type is not None:
                act_func = \
                    get_activation_func(self._activation_type)
                res = \
                    act_func(res, name='activation')

            if self._normalizer_type is not None:
                normalizer = get_normalizer(self._normalizer_type,
                    train=self._train)
                res = \
                    normalizer(res, 'normalizer')

        return tf.reshape(res,
            tf.concat([output_shape[:-1], tf.constant([self.hidden_size])], axis=0)
        )

class DenseEmbedding(object):
    def __init__(self, input_depth, hidden_size, scope,
        init_matrix, seed=1):

        self._scope = scope
        self.input_depth = input_depth
        self.hidden_size = hidden_size
        self.seed = seed

        with tf.variable_scope(self._scope):
            self.weight = tf.Variable(init_matrix, dtype=tf.float32)

        self.initialize_op = tf.initialize_variables([self.weight])

    def __call__(self, input_vec):
        output_shape = tf.shape(input_vec)

        with tf.variable_scope(self._scope):
            res = tf.nn.embedding_lookup(self.weight, input_vec)

        return tf.reshape(res,
            tf.concat([output_shape, tf.constant([self.hidden_size])], axis=0)
        )


def dummy_step(cell, _input, _state, _output_tensor, _i):
    _output, _next_state = cell(_input[:, _i], _state[:, _i])

    _state = tf.concat(
        [_state, tf.expand_dims(_next_state, 1)], axis=1
    )

    _output_tensor = tf.concat(
        [_output_tensor, tf.expand_dims(_output, 1)], axis=1
    )
    _i += 1

    return _input, _state, _output_tensor, _i

def _condition(_input, _state, _output_tensor, _i):
    return _i < tf.shape(_input)[1]

def _dynamic_rnn(cell, input, initial_state, hidden_size, use_lstm=True):
    _batch_size = tf.shape(input)[0]
    _dummy_outputs = tf.zeros(
        [_batch_size, 1, hidden_size]
    )

    if initial_state is None:
        if use_lstm:
            initial_state = tf.zeros(
                [_batch_size, 1, 2 * hidden_size]
            )
        else:
            initial_state = tf.zeros(
                [_batch_size, 1, hidden_size]
            )

    # specify first two
    _lambda = lambda a, b, c, d: dummy_step(cell, a, b, c, d)

    if use_lstm:
        _, _rnn_state_arr, _rnn_output_arr, _ = tf.while_loop(
            _condition, _lambda,
            [input, initial_state, _dummy_outputs, tf.constant(0)],
            shape_invariants=[input.get_shape(),
                tf.TensorShape((None, None, 2 * hidden_size)),
                tf.TensorShape((None, None, hidden_size)),
                tf.TensorShape(())]
        )

    else:
        _, _rnn_state_arr, _rnn_output_arr, _ = tf.while_loop(
            _condition, _lambda,
            [input, initial_state, _dummy_outputs, tf.constant(0)],
            shape_invariants=[input.get_shape(),
                tf.TensorShape((None, None, hidden_size)),
                tf.TensorShape((None, None, hidden_size)),
                tf.TensorShape(())]
        )

    _rnn_state_arr = _rnn_state_arr[:, 1:]
    _rnn_output_arr = _rnn_output_arr[:, 1:]
    return _rnn_output_arr, _rnn_state_arr