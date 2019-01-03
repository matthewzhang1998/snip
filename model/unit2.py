import tensorflow as tf
from model.networks.rnn_unit2 import *
from util.network_util import *

def get_model(name):
    name = name.lower()
    if name == 'mlp':
        raise NotImplementedError

    elif name == 'rnn':
        return RNNModel

    elif name == 'cnn':
        raise NotImplementedError

class Unit(object):
    def __init__(self, scope, params, input_size, num_classes, seed=1, init=None):
        self.params = params
        self.Snip = {}
        self.scope = scope
        with tf.variable_scope(scope):
            self.model = get_model(params.model_type)(
                params, input_size, num_classes, seed, init
            )
        self.Tensor = {}
        self.Op = {}

    def build_sparse(self, sparse_list):
        with tf.variable_scope(self.scope):
            self.model.build_sparse(sparse_list)
            self.initialize_op = self.model.initialize_op

    def run(self, features):
        with tf.variable_scope(self.scope):
            self.Tensor['Predictions'] = self.model(features)
            return self.Tensor['Predictions']

    def unit(self, minibatch, loss_fnc, embed=None):
        with tf.variable_scope(self.scope):
            self.Tensor['Unit_Pred'] = self.model.unit(
                minibatch['Features']
            )

            self.Snip['Dummy_Kernel'], self.Snip['Dummy_Bias'] = \
                self.model.get_dummy_variables()

            if embed != None:
                self.Tensor['Unit_Pred'] = tf.einsum('ijk,kl->ijl',
                    self.Tensor['Unit_Pred'], embed[0]) + embed[1]

            self.Tensor['Unit_Loss'] = tf.reduce_mean(loss_fnc(
                self.Tensor['Unit_Pred'], minibatch['Labels']
            ))

            # one layer case
            print(self.Snip['Dummy_Kernel'])
            self.Tensor['Unit_Grad'] = tf.gradients(
                self.Tensor['Unit_Loss'], self.Snip['Dummy_Kernel'][0])*self.Snip['Dummy_Kernel'][0]
