import tensorflow as tf
from model.networks.rnn_unit import *
from util.network_util import *

def get_model(name):
    name = name.lower()
    if name == 'mlp':
        return MLPUnitModel

    elif name == 'rnn':
        return RNNUnitModel

    elif name == 'cnn':
        return CNNUnitModel

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

    def run(self, features):
        with tf.variable_scope(self.scope):
            self.Tensor['Predictions'] = self.model(features)
            return self.Tensor['Predictions']

    def prune(self, minibatch, loss_fnc, embed=None):
        with tf.variable_scope(self.scope):
            self.Tensor['Snip_Pred'] = self.model.unitwise(
                minibatch['Features']
            )

            self.Snip['Weight'] = self.model.weight_variables()
            self.Snip['Mask'] = self.model.get_mask()
            self.Snip['Comb'] = self.model.get_weighted_mask()
            self.Placeholder = self.Snip['Mask']

            if embed != None:
                self.Tensor['Snip_Pred'] = tf.einsum('ijk,kl->ijl',
                    self.Tensor['Snip_Pred'], embed[0]) + embed[1]

            self.Tensor['Snip_Loss'] = tf.reduce_mean(loss_fnc(
                self.Tensor['Snip_Pred'], minibatch['Labels']
            ))

            self.Tensor['Snip_Grad'] = tf.gradients(
                self.Tensor['Snip_Loss'], self.Snip[self.params.grad_param])
            ratio = tf.constant(1-self.params.prune_k, tf.float32)

            self.Tensor['Snip_Index'] = []
            return self.Tensor['Snip_Index']

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

            self.Tensor['Unit_Grad'] = tf.gradients(
                self.Tensor['Unit_Loss'], self.Snip['Dummy_Kernel'])*self.Snip['Dummy_Kernel'][0]
