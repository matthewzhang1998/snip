import tensorflow as tf

from tensorflow.python.ops.rnn_cell_impl import *
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.layers import base as base_layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.deprecation import deprecated

from util.network_util import *

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"

def init(shape=None, name=None, init_method=None, dtype=tf.float32,
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

    elif init_method == 'variance_scaling':
        initializer = tf.contrib.layers.variance_scaling_initializer(
            factor=init_para['factor'], uniform=init_para['uniform'],
            seed=seed, dtype=dtype
        )

    elif init_method == 'bradly':
        initializer = bradly_initializer(
            shape, alpha=init_para['alpha'],
            seed=seed, dtype=dtype
        )

    else:
        raise ValueError("Unsupported initialization method!")

    return initializer

class BasicMaskRNNCell(LayerRNNCell):
  """The most basic RNN cell.
  Note that this cell is not optimized for performance. Please use
  `tf.contrib.cudnn_rnn.CudnnRNNTanh` for better performance on GPU.
  Args:
    num_units: int, The number of units in the RNN cell.
    activation: Nonlinearity to use.  Default: `tanh`. It could also be string
      that is within Keras activation function names.
    reuse: (optional) Python boolean describing whether to reuse variables
     in an existing scope.  If not `True`, and the existing scope already has
     the given variables, an error is raised.
    name: String, the name of the layer. Layers with the same name will
      share weights, but to avoid mistakes we require reuse=True in such
      cases.
    dtype: Default dtype of the layer (default of `None` means use the type
      of the first input). Required when `build` is called before `call`.
    **kwargs: Dict, keyword named properties for common layer attributes, like
      `trainable` etc when constructing the cell from configs of get_config().
  """

  @deprecated(None, "This class is equivalent as tf.keras.layers.SimpleRNNCell,"
                    " and will be replaced by that in Tensorflow 2.0.")
  def __init__(self,
               num_units,
               activation=None,
               reuse=None,
               name=None,
               dtype=None,
               input_depth=None,
               init_data=None,
               seed=12345,
               **kwargs):
    super(BasicMaskRNNCell, self).__init__(
        _reuse=reuse, name=name, dtype=dtype, **kwargs)
    if context.executing_eagerly() and context.num_gpus() > 0:
      logging.warn("%s: Note that this cell is not optimized for performance. "
                   "Please use tf.contrib.cudnn_rnn.CudnnRNNTanh for better "
                   "performance on GPU.", self)

    # Inputs must be 2-dimensional.
    self.input_spec = base_layer.InputSpec(ndim=2)

    self._num_units = num_units
    if activation:
      self._activation = activations.get(activation)
    else:
      self._activation = math_ops.tanh

    self._mask = tf.placeholder(dtype=tf.float32,
        shape=[input_depth + self._num_units, self._num_units], name='mask')

    #initializer = init([input_depth + self._num_units, self._num_units], 'kernel',
    #    init_data['w_init_method'], tf.float32, init_data['w_init_para'],
    #    seed=seed, trainable=True
    #)
    self._kernel = self.add_variable(
        _WEIGHTS_VARIABLE_NAME,
        shape=[input_depth + self._num_units, self._num_units],
        #initializer=initializer
    )

    self._combined = self._kernel * self._mask

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  @tf_utils.shape_type_conversion
  def build(self, inputs_shape):
    if inputs_shape[-1] is None:
      raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                       % str(inputs_shape))

    self._bias = self.add_variable(
        _BIAS_VARIABLE_NAME,
        shape=[self._num_units],
        initializer=init_ops.zeros_initializer(dtype=self.dtype))

    self.built = True

  def call(self, inputs, state):
    """Most basic RNN: output = new_state = act(W * input + U * state + B)."""

    gate_inputs = math_ops.matmul(
        array_ops.concat([inputs, state], 1), self._combined)
    gate_inputs = nn_ops.bias_add(gate_inputs, self._bias)
    output = self._activation(gate_inputs)
    return output, output

  def get_config(self):
    config = {
        "num_units": self._num_units,
        "activation": activations.serialize(self._activation),
        "reuse": self._reuse,
    }
    base_config = super(BasicMaskRNNCell, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

class BasicMaskLSTMCell(LayerRNNCell):
    pass
