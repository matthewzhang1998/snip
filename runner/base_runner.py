import tensorflow as tf
from model.random import *
from model.snip import *

from util.logger_util import *

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

        self.Writer = tf.summary.FileWriter(self.Dir, self.Sess.graph)

    def preprocess(self):
        raise NotImplementedError

    def train(self, i):
        raise NotImplementedError

    def val(self, i):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError