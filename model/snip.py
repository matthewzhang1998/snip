import tensorflow as tf
from util.random_util import *
from model.base import *

class Snip(Pruner):
    def __init__(self, scope, params, input_size, num_classes):
        super(Snip, self).__init__(scope, params, input_size, num_classes)

        self.Tensor['Delta'] = {}

        with tf.variable_scope(scope):
            for weight in self.weight_params:
                self.Tensor['Delta'][weight.name] = gaussian_noise_layer(
                    tf.ones_like(weight), self.params.noise_delta
                )

        self.Op['Delta'] = \
            self.model.set_weighted_params(self.Tensor['Delta'], 'Delta')

    def prune(self, minibatch, loss_fnc):

        self.Tensor['Snip_Pred'] = self.model.weighted(
            minibatch['Features'], 'Delta'
        )

        self.Tensor['Snip_Loss'] = tf.reduce_mean(loss_fnc(
            self.Tensor['Snip_Pred'], minibatch['Labels']
        ))

        keys, vars = zip(*self.model.Mask['Delta'].items())

        self.Tensor['Snip_Grad'] = tf.gradients(
            self.Tensor['Snip_Loss'], vars)

        self.Tensor['Mask'] = {}

        ratio = tf.constant(1-self.params.prune_k, tf.float32)

        for ix, grad in enumerate(self.Tensor['Snip_Grad']):
            print(grad)
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

            u_ix = tf.unravel_index(k_ix, old_shape)

            u_ix = tf.transpose(u_ix, [1,0])

            scatter_ones = tf.cast(tf.ones_like(k_ix), tf.float32)

            mask = tf.scatter_nd(u_ix, scatter_ones, old_shape)
            name = keys[ix]
            self.Tensor['Mask'][name] = mask

        self.Op['Sparse'] = \
            self.model.set_weighted_params(self.Tensor['Mask'], 'Sparse')

    def run(self, features):
        self.Tensor['predictions'] = self.model.weighted(features, 'Sparse')
        return self.Tensor['predictions']
