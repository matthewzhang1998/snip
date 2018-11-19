import tensorflow as tf
import numpy as np
from util import norm_util


def get_activation_func(activation_type):
    if activation_type == 'leaky_relu':
        activation_func = tf.nn.leaky_relu
    elif activation_type == 'tanh':
        activation_func = tf.nn.tanh
    elif activation_type == 'relu':
        activation_func = tf.nn.relu
    elif activation_type == 'elu':
        activation_func = tf.nn.elu
    elif activation_type == 'none':
        activation_func = tf.identity
    elif activation_type == 'sigmoid':
        activation_func = tf.sigmoid
    else:
        raise ValueError(
            "Unsupported activation type: {}!".format(activation_type)
        )
    return activation_func

def get_rnn_cell(rnn_cell_type):
    cell_args = {}
    if rnn_cell_type == 'basic':
        cell_type = tf.contrib.rnn.BasicRNNCell
    elif rnn_cell_type == 'lstm':
        cell_type = tf.contrib.rnn.BasicLSTMCell
        cell_args['state_is_tuple'] = False
    elif rnn_cell_type == 'gru':
        cell_type = tf.contrib.rnn.GRUCell
    elif rnn_cell_type == 'norm_lstm':
        cell_type = tf.contrib.rnn.LayerNormBasicLSTMCell
    else:
        raise ValueError("Unsupported cell type: {}".format(rnn_cell_type))
    return cell_type, cell_args


def get_normalizer(normalizer_type, train=True):
    if normalizer_type == 'batch':
        normalizer = norm_util.batch_norm_with_train if train else \
            norm_util.batch_norm_without_train

    elif normalizer_type == 'layer':
        normalizer = norm_util.layer_norm

    elif normalizer_type == 'none':
        normalizer = tf.identity

    else:
        raise ValueError(
            "Unsupported normalizer type: {}!".format(normalizer_type)
        )
    return normalizer


def normc_initializer(shape, seed=1234, stddev=1.0, dtype=tf.float32):
    npr = np.random.RandomState(seed)
    out = npr.randn(*shape).astype(np.float32)
    out *= stddev / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
    return tf.constant(out)


def weight_variable(shape, name, init_method=None, dtype=tf.float32,
                    init_para=None, seed=1234, trainable=True):
    """ @brief:
            Initialize weights

        @input:
            shape: list of int, shape of the weights
            init_method: string, indicates initialization method
            init_para: a dictionary,
            init_val: if it is not None, it should be a tensor

        @output:
            var: a TensorFlow Variable
    """

    if init_method is None or init_method == 'zero':
        initializer = tf.zeros_initializer(shape, dtype=dtype)

    if init_method == "normc":
        print(shape)
        var = normc_initializer(
            shape, stddev=init_para['stddev'],
            seed=seed, dtype=dtype
        )
        return tf.get_variable(initializer=var, name=name, trainable=trainable)

    elif init_method == "normal":
        initializer = tf.random_normal_initializer(
            mean=init_para["mean"], stddev=init_para["stddev"],
            seed=seed, dtype=dtype
        )

    elif init_method == "truncated_normal":
        initializer = tf.truncated_normal_initializer(
            mean=init_para["mean"], stddev=init_para["stddev"],
            seed=seed, dtype=dtype
        )

    elif init_method == "uniform":
        initializer = tf.random_uniform_initializer(
            minval=init_para["minval"], maxval=init_para["maxval"],
            seed=seed, dtype=dtype
        )

    elif init_method == "constant":
        initializer = tf.constant_initializer(
            value=init_para["val"], dtype=dtype
        )

    elif init_method == "xavier":
        initializer = tf.contrib.layers.xavier_initializer(
            uniform=init_para['uniform'], seed=seed, dtype=dtype
        )

    elif init_method == 'orthogonal':
        initializer = tf.orthogonal_initializer(
            gain=1.0, seed=seed, dtype=dtype
        )

    else:
        raise ValueError("Unsupported initialization method!")

    var = tf.get_variable(initializer=initializer(shape),
                          name=name, trainable=trainable)

    return var


class Linear(object):
    """
    Performs a simple weight multiplication without activations
    """

    def __init__(self, dims, scope, train,
                 normalizer_type, init_data,
                 dtype=tf.float32, reuse=None):

        self._scope = scope
        self._num_layer = len(dims) - 1  # the last one is the input dim
        self._w = [None] * self._num_layer
        self._b = [None] * self._num_layer
        self._train = train
        self._reuse = reuse

        self._normalizer_type = normalizer_type
        self._init_data = init_data
        self._last_dim = dims[-1]

        # initialize variables
        with tf.variable_scope(scope, reuse=self._reuse):
            for ii in range(self._num_layer):
                with tf.variable_scope("layer_{}".format(ii)):
                    dim_in, dim_out = dims[ii], dims[ii + 1]

                    self._w[ii] = weight_variable(
                        shape=[dim_in, dim_out], name='weights',
                        init_method=self._init_data[ii]['w_init_method'],
                        init_para=self._init_data[ii]['w_init_para'],
                        dtype=dtype, trainable=self._train
                    )

                    self._b[ii] = weight_variable(
                        shape=[dim_out], name='bias',
                        init_method=self._init_data[ii]['b_init_method'],
                        init_para=self._init_data[ii]['b_init_para'],
                        dtype=dtype, trainable=self._train
                    )

    def __call__(self, input_vec):
        self._h = [None] * self._num_layer

        output_shape = tf.shape(input_vec)
        flat_input = tf.reshape(input_vec,
                                tf.concat([[-1], [output_shape[-1]]], axis=0)
                                )

        with tf.variable_scope(self._scope, reuse=self._reuse):
            for ii in range(self._num_layer):
                with tf.variable_scope("layer_{}".format(ii)):
                    if ii == 0:
                        self._h[ii] = tf.matmul(flat_input, self._w[ii]) \
                                      + self._b[ii]

                    else:
                        self._h[ii] = \
                            tf.matmul(self._h[ii - 1], self._w[ii]) \
                            + self._b[ii]

                    if self._normalizer_type[ii] is not None:
                        normalizer = get_normalizer(self._normalizer_type[ii],
                                                    train=self._train)
                        self._h[ii] = \
                            normalizer(self._h[ii], 'normalizer_' + str(ii))

        return tf.reshape(self._h[-1],
                          tf.concat([output_shape[:-1], [self._last_dim]], axis=0)
                          )

    def weights(self):
        weights = []
        with tf.variable_scope(self._scope, reuse=self._reuse):
            for ii in range(self._num_layer):
                with tf.variable_scope("layer_{}".format(ii)):
                    weights.append(self._w[ii])
        return weights

class Recurrent_Network(object):
    """
    borrows tensorflow api for recurrent networks
    """

    def __init__(self, scope, activation_type,
                 normalizer_type, recurrent_cell_type,
                 train, hidden_size, dtype=tf.float32, reuse=None):
        self._scope = scope
        _cell_proto, _cell_kwargs = get_rnn_cell(recurrent_cell_type)
        self._activation_type = activation_type
        self._normalization_type = normalizer_type
        self._train = train
        self._reuse = reuse
        self._hidden_size = hidden_size
        with tf.variable_scope(scope):
            self._cell = _cell_proto(hidden_size, **_cell_kwargs)

    def __call__(self, input_tensor, hidden_states=None):
        with tf.variable_scope(self._scope, reuse=self._reuse):
            _rnn_outputs, _rnn_states = tf.nn.dynamic_rnn(
                self._cell, input_tensor, initial_state=hidden_states,
                dtype=tf.float32
            )
            if self._activation_type is not None:
                act_func = \
                    get_activation_func(self._activation_type)
                _rnn_outputs = \
                    act_func(_rnn_outputs, name='activation_0')

            if self._normalization_type is not None:
                normalizer = get_normalizer(self._normalization_type,
                                            train=self._train)
                _rnn_outputs = \
                    normalizer(_rnn_outputs, 'normalizer_0')
        return _rnn_outputs, _rnn_states

    def weights(self):
        with tf.variable_scope(self._scope, reuse=self._reuse):
            pass

class Bidirectional_Recurrent_Network(object):
    """
    borrows tensorflow api for recurrent networks
    """

    def __init__(self, scope, activation_type,
                 normalizer_type, recurrent_cell_type,
                 train, hidden_size, dtype=tf.float32, reuse=None):
        self._scope = scope
        _cell_proto, _cell_kwargs = get_rnn_cell(recurrent_cell_type)
        self._activation_type = activation_type
        self._normalization_type = normalizer_type
        self._train = train
        self._reuse = reuse
        self._hidden_size = hidden_size
        with tf.variable_scope(scope):
            self._cell = _cell_proto(hidden_size, **_cell_kwargs)

    def __call__(self, input_tensor, fw_states=None, bw_states=None):
        with tf.variable_scope(self._scope, reuse=self._reuse):
            _rnn_outputs, _rnn_states = tf.nn.bidirectional_dynamic_rnn(
                self._cell, input_tensor, initial_state_fw=fw_states,
                initial_state_bw=bw_states, dtype=tf.float32
            )

            _fw, _bw = _rnn_outputs

            if self._activation_type[0] is not None:
                act_func = \
                    get_activation_func(self._activation_type[0])
                _fw = \
                    act_func(_fw, name='activation_0')
                _bw = \
                    act_func(_bw, name='activation_0')

            if self._normalization_type is not None:
                normalizer = get_normalizer(self._normalization_type[0],
                                            train=self._train)
                _fw = \
                    normalizer(_fw, 'normalizer_0')
                _bw = \
                    normalizer(_bw, 'normalizer_0')

        return (_fw, _bw), _rnn_states


class Dilated_Recurrent_Network(object):
    """
    borrows tensorflow api for recurrent networks
    """

    def __init__(self, scope, activation_type,
                 normalizer_type, recurrent_cell_type,
                 train, hidden_size, dtype=tf.float32,
                 dilation=1, reuse=None):
        self._scope = scope
        _cell_proto, _cell_kwargs = get_rnn_cell(recurrent_cell_type)
        self._activation_type = activation_type
        self._normalization_type = normalizer_type
        self._train = train
        self._dilation = dilation
        self._hidden_size = hidden_size
        self._reuse = reuse
        with tf.variable_scope(scope):
            self._cell = _cell_proto(hidden_size, **_cell_kwargs)

    def __call__(self, input_tensor, hidden_states):

        _hidden_states = tf.expand_dims(hidden_states, 1)
        _dilated_hidden_states = tf.tile(
            _hidden_states, [1, self._dilation, 1]
        )
        _batch_size = tf.shape(input_tensor)[0]
        _dummy_outputs = tf.zeros(
            [_batch_size, 1, self._hidden_size]
        )

        with tf.variable_scope(self._scope, reuse=self._reuse):
            # construct dilated rnn
            def _dilated_rnn(_input, _state, _output_tensor, _i):
                _output, _next_state = self._cell(_input[:, _i],
                                                  _state[:, _i % self._dilation])

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

            _, _rnn_state_arr, _rnn_output_arr, _ = tf.while_loop(
                _condition, _dilated_rnn,
                [input_tensor, _dilated_hidden_states,
                 _dummy_outputs, tf.constant(0)],
                shape_invariants=[input_tensor.get_shape(),
                                  tf.TensorShape((None, None, self._hidden_size)),
                                  tf.TensorShape((None, None, self._hidden_size)),
                                  tf.TensorShape(())]
            )

            _rnn_states = _rnn_state_arr[:, -1]
            _rnn_outputs = _rnn_output_arr[:, 1:]

            _rnn_outputs, _rnn_states = tf.nn.dynamic_rnn(
                self._cell, input_tensor, initial_state=hidden_states
            )
            if self._activation_type[0] is not None:
                act_func = \
                    get_activation_func(self._activation_type[0])
                _rnn_outputs = \
                    act_func(_rnn_outputs, name='activation_0')

            if self._normalization_type is not None:
                normalizer = get_normalizer(self._normalization_type[0],
                                            train=self._train)
                _rnn_outputs = \
                    normalizer(_rnn_outputs, 'normalizer_0')
        return _rnn_outputs, _rnn_states


class MLP(object):
    """ Multi Layer Perceptron (MLP)
                Note: the number of layers is N

        Input:
                dims: a list of N+1 int, number of hidden units (last one is the
                output dimension)
                act_func: a list of N activation functions
                add_bias: a boolean, indicates whether adding bias or not
                scope: tf scope of the model

    """

    def __init__(self, dims, scope, train,
                 activation_type, normalizer_type, init_data,
                 dtype=tf.float32, reuse=None):

        self._scope = scope
        self._num_layer = len(dims) - 1  # the last one is the output dim
        self._p = [None] * self._num_layer
        self._w = {'default': [None] * self._num_layer}
        self._b = [None] * self._num_layer
        self._train = train
        self._last_dim = dims[-1]
        self._reuse = reuse

        self._activation_type = activation_type
        self._normalizer_type = normalizer_type
        self._init_data = init_data

        # initialize variables
        with tf.variable_scope(scope, reuse=reuse):
            for ii in range(self._num_layer):
                with tf.variable_scope("layer_{}".format(ii)):
                    dim_in, dim_out = dims[ii], dims[ii + 1]

                    self._p[ii] = weight_variable(
                        shape=[dim_in, dim_out], name='permanent_weights',
                        init_method=self._init_data[ii]['w_init_method'],
                        init_para=self._init_data[ii]['w_init_para'],
                        dtype=dtype, trainable=self._train
                    )

                    self._w['default'][ii] = tf.identity(self._p[ii])

                    self._b[ii] = weight_variable(
                        shape=[dim_out], name='bias',
                        init_method=self._init_data[ii]['b_init_method'],
                        init_para=self._init_data[ii]['b_init_para'],
                        dtype=dtype, trainable=self._train
                    )

    def __call__(self, input_vec, scope='default'):
        self._h = [None] * self._num_layer

        output_shape = tf.shape(input_vec)
        flat_input = tf.reshape(input_vec,
                                tf.concat([[-1], [output_shape[-1]]], axis=0)
                                )

        with tf.variable_scope(self._scope, reuse=self._reuse):
            for ii in range(self._num_layer):
                with tf.variable_scope("layer_{}".format(ii),
                                       reuse=tf.AUTO_REUSE):
                    if ii == 0:
                        self._h[ii] = tf.matmul(flat_input, self._w[scope][ii]) \
                                      + self._b[ii]

                    else:
                        self._h[ii] = \
                            tf.matmul(self._h[ii - 1], self._w[scope][ii]) \
                            + self._b[ii]

                    if self._activation_type[ii] is not None:
                        act_func = \
                            get_activation_func(self._activation_type[ii])
                        self._h[ii] = \
                            act_func(self._h[ii], name='activation_' + str(ii))

                    if self._normalizer_type[ii] is not None:
                        normalizer = get_normalizer(self._normalizer_type[ii],
                                                    train=self._train)
                        self._h[ii] = \
                            normalizer(self._h[ii], 'normalizer_' + str(ii))

        return tf.reshape(self._h[-1],
            tf.concat([output_shape[:-1], [self._last_dim]], axis=0)
        )

    def weights(self):
        weights = [self._p[ii] for ii in range(self._num_layer)]
        return weights

    def assign_masks(self, param_dict, scope):
        self._w[scope] = [None] * self._num_layer
        for ii in range(self._num_layer):
            name = self._p[ii].name
            self._w[scope][ii] = param_dict[name] * self._p[ii]