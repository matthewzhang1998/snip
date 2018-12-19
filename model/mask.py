import tensorflow as tf
from model.base import *
from util.network_util import *

class Mask(Pruner):
    def __init__(self, scope, params, input_size, num_classes, seed=1, init=None):
        super(Mask, self).__init__(scope, params, input_size, num_classes, seed, init)
        self.Snip['Mask'] = self.model.get_mask()
        self.Snip['Comb'] = self.model.get_weighted_mask()

        self.Placeholder = self.Snip['Mask']

    def run(self, features):
        self.Tensor['Predictions'] = self.model(features)
        return self.Tensor['Predictions']

    def prune(self, minibatch, loss_fnc):
        self.Tensor['Snip_Pred'] = self.model(
            minibatch['Features']
        )

        self.Tensor['Snip_Loss'] = tf.reduce_mean(loss_fnc(
            self.Tensor['Snip_Pred'], minibatch['Labels']
        ))

        self.Tensor['Snip_Grad'] = tf.gradients(
            self.Tensor['Snip_Loss'], self.Snip[self.params.grad_param])
        ratio = tf.constant(1-self.params.prune_k, tf.float32)

        self.Tensor['Snip_Index'] = []

        for ix, grad in enumerate(self.Tensor['Snip_Grad']):
            old_shape = tf.shape(grad)
            flat_grad = tf.reshape(tf.abs(grad), [-1])

            n_params = tf.cast(
                tf.reshape(tf.reduce_prod(old_shape), [-1]), tf.float32
            )

            prune_ratio = tf.reshape(
                tf.cast(tf.round(n_params * ratio), tf.int32), []
            )

            # top gradients per weight layer
            k_ix = tf.nn.top_k(flat_grad, prune_ratio)[1]

            self.Tensor['Snip_Index'].append(k_ix)
        return self.Tensor['Snip_Index']

class MaskCopy(Mask):
    def __init__(self, *args, target=None):
        super(MaskCopy, self).__init__(*args)
        self.model.Network['Linear'] = MLPCopy(target.model.Network['Linear'], self.scope)

        self.Snip['Mask'] = self.model.get_mask()
        self.Snip['Comb'] = self.model.get_weighted_mask()
        self.Snip['Weight'] = self.model.weight_variables()

class MaskTrain(object):
    def __init__(self, scope, params, input_size, num_classes, seed=1, init=None):
        self.params = params
        self.Snip = {}
        self.scope = scope
        with tf.variable_scope(scope):
            self.model = get_model(params.model_type)(
                params, input_size, num_classes, seed, init
            )
            self.Snip['Weight'] = self.model.weight_variables()
        self.Tensor = {}
        self.Op = {}

        self.Snip['Mask'] = self.model.get_mask()
        self.Snip['Mask_Train'] = self.model.get_mask_train()
        self.Snip['Comb'] = self.model.get_weighted_mask()

        self.Placeholder = self.Snip['Mask']

    def run(self, features):
        self.Tensor['Predictions'] = self.model(features)
        return self.Tensor['Predictions']

    def prune(self, minibatch, loss_fnc):
        self.Tensor['Snip_Pred'] = self.model.train_mask(
            minibatch['Features']
        )
        self.Snip['Comb_Train'] = self.model.get_weighted_mask_train()

        self.Tensor['Snip_Loss'] = tf.reduce_mean(loss_fnc(
            self.Tensor['Snip_Pred'], minibatch['Labels']
        ))

        self.Tensor['Snip_Grad'] = tf.gradients(
            self.Tensor['Snip_Loss'], self.Snip['Comb_Train'])

def get_model(name):
    name = name.lower()
    if name == 'mlp':
        return MLPTrainMaskModel

    elif name == 'rnn':
        return RNNTrainMaskModel

    elif name == 'cnn':
        return CNNTrainMaskModel

