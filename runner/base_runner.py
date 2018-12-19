import tensorflow as tf
from model.random import *
from model.snip import *

from util.logger_util import *

def SoftmaxCE(logits, labels):
    return tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=labels
    )

def MSELoss(logits, labels):
    return tf.losses.mean_squared_error(labels, logits)

class BaseRunner(object):
    def __init__(self, scope, params):
        self.params = params
        self.scope = scope

        self.Tensor = {}
        self.Model = {}
        self.Placeholder = {}
        self.Output = {}
        self.Summary = {}

        self.Sess = tf.Session()

        self.Dir = get_dir(params.log_dir)

    def preprocess(self):
        raise NotImplementedError

    def train(self, i):
        raise NotImplementedError

    def val(self, i):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError