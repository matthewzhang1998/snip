import tensorflow as tf
import numpy as np
from model.networks.mlp import *
# from model.networks.cnn import *
# from model.networks.rnn import *


class Pruner(object):
    def __init__(self, scope, params, input_size, num_classes):
        self.params = params
        with tf.variable_scope(scope):
            self.model = get_model(params.model_type)(
                params, input_size, num_classes
            )
            self.weight_params = [var for var in
                self.model.weight_variables()]
        self.Tensor = {}
        self.Op = {}

def get_model(name):
    name = name.lower()
    if name == 'mlp':
        return MLPModel

    elif name == 'rnn':
        return RNNModel

    elif name == 'cnn':
        return CNNModel