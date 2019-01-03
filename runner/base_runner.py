import tensorflow as tf
from model.random import *
from model.snip import *

from util.logger_util import *

def SoftmaxCE(logits, labels):
    return tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=labels
    )

def SoftmaxSliceCE(logits, labels):
    return tf.nn.softmax_cross_entropy_with_logits(
        logits=logits[:,-1], labels=labels
    )

def MSELoss(logits, labels):
    return tf.losses.mean_squared_error(labels, logits)

def Seq2SeqLoss(logits, labels):
    return tf.contrib.seq2seq.sequence_loss(
        logits, labels, tf.ones_like(logits[:,:,0], dtype=tf.float32),
        average_across_timesteps=False,
        average_across_batch=True
    )

class BaseRunner(object):
    def __init__(self, scope, params):
        self.params = params
        self.scope = scope

        self.Tensor = {}
        self.Model = {}
        self.Placeholder = {}
        self.Output = {}
        self.Summary = {}

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth=True
        self.Sess = tf.Session(config=config)

        self.Dir = get_dir(params.log_dir)

    def preprocess(self):
        raise NotImplementedError

    def train(self, i):
        raise NotImplementedError

    def val(self, i):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError