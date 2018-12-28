import numpy as np
import tensorflow as tf
from model.mask import *
from runner.base_runner import *
from util.optimizer_util import *
from tensorflow import keras

from data.load_pen import *

ZERO_32 = tf.constant(0.0, dtype=tf.float32)

def get_mnist_dataset():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train = np.concatenate([x_train, x_test[:3000]], axis=0)
    y_train = np.concatenate([y_train, y_test[:3000]], axis=0)
    x_test = x_test[3000:]
    y_test = y_test[3000:]

    x_train /= 255
    x_test /= 255

    num_classes = 10
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    x_train = np.reshape(x_train, [-1, 28, 28])
    x_test = np.reshape(x_test, [-1, 28, 28])
    return (x_train, y_train), (x_test, y_test)

class MNISTRunner(BaseRunner):
    def __init__(self, scope, params):
        super(MNISTRunner, self).__init__(scope, params)

        self.Mask = {}

        self.Data = get_mnist_dataset()
        self._build_outputs()
        self._build_summary()
        self.Writer = {}

        self.Saver = tf.train.Saver()

        self._npr = np.random.RandomState(params.seed)

        self.learning_rate = params.learning_rate
        self.pretrain_learning_rate = params.pretrain_learning_rate

        self.Writer['Random'] = \
            tf.summary.FileWriter(self.Dir+'/random', self.Sess.graph)
        self.Writer['Snip'] = \
            tf.summary.FileWriter(self.Dir+'/snip', self.Sess.graph)

    def _build_outputs(self):
        with tf.variable_scope(self.scope):

            self.Model['Snip'] = Mask('snip', self.params,
                28, 10, self.params.seed)
            self.Model['Random'] = Mask('random', self.params,
                28, 10, self.params.seed)

            self.start_ix = 0

            self.Placeholder['Input_Feature'] = tf.placeholder(
                shape=[None, None, 28], dtype=tf.float32,
            )

            self.Placeholder['Learning_Rate'] = tf.placeholder(
                tf.float32, []
            )

            self.Placeholder['Input_Label'] = tf.placeholder(
                tf.int32, [None, 10]
            )

            self.Placeholder['Input_Logits'] = tf.placeholder(tf.float32,
                [None, 10])

            self.Tensor['Proto_Minibatch'] = {
                'Features': self.Placeholder['Input_Feature'],
                'Labels': self.Placeholder['Input_Label']
            }

            self.Tensor['Loss_Function'] = \
                SoftmaxCE

            self.Tensor['Recurrent_Loss_Function'] = \
                SoftmaxSliceCE

            self.Output['Optimizer'] = get_optimizer(
                self.params, self.Placeholder['Learning_Rate']
            )

            self.Model['Snip'].prune(
                self.Tensor['Proto_Minibatch'], self.Tensor['Recurrent_Loss_Function']
            )
            self.Model['Random'].prune(
                self.Tensor['Proto_Minibatch'], self.Tensor['Recurrent_Loss_Function']
            )

            self.Placeholder['Random_Mask'] = self.Model['Random'].Placeholder
            self.Placeholder['Snip_Mask'] = self.Model['Snip'].Placeholder

            self.Tensor['Snip_Grad'] = self.Model['Snip'].Tensor['Snip_Grad']
            self.Tensor['Random_Grad'] = self.Model['Random'].Tensor['Snip_Grad']

            self.Output['Random_Pred'] = self.Model['Random'].run(
                self.Placeholder['Input_Feature']
            )[:,-1]

            self.Output['Snip_Pred'] = self.Model['Snip'].run(
                self.Placeholder['Input_Feature']
            )[:,-1]

            self.Output['Random_Loss'] = tf.reduce_mean(
                self.Tensor['Loss_Function'](
                    self.Output['Random_Pred'], self.Placeholder['Input_Label']
                )
            )
            self.Output['Snip_Loss'] = tf.reduce_mean(
                self.Tensor['Loss_Function'](
                    self.Output['Snip_Pred'], self.Placeholder['Input_Label']
                )
            )

            self.Output['Random_Loss'] += self.params.weight_decay * \
                tf.reduce_sum(
                    [tf.nn.l2_loss(t) for t in self.Model['Random'].Snip['Weight']]
                )

            self.Output['Snip_Loss'] += self.params.weight_decay * \
                tf.reduce_sum(
                    [tf.nn.l2_loss(t) for t in self.Model['Snip'].Snip['Weight']]
                )

            self.Output['Random_Train'] = \
                self.Output['Optimizer'].minimize(self.Output['Random_Loss'])
            self.Output['Snip_Train'] = \
                self.Output['Optimizer'].minimize(self.Output['Snip_Loss'])

    def _build_summary(self):
        self.Output['Loss'] = tf.reduce_mean(
            self.Tensor['Loss_Function'](
                self.Placeholder['Input_Logits'],
                self.Placeholder['Input_Label']
            )
        )
        self.Output['Round'] = \
            tf.argmax(self.Placeholder['Input_Logits'], 1)

        self.Output['Error'] = 1 - tf.reduce_mean(
            tf.cast(tf.equal(
                self.Output['Round'],
                tf.argmax(self.Placeholder['Input_Label'], 1)
            ), tf.float32)
        )

        self.Summary['Train_Error'] = tf.summary.scalar(
            'Train_Error', self.Output['Error']
        )
        self.Summary['Val_Error'] = tf.summary.scalar(
            'Val_Error', self.Output['Error']
        )

        self.Summary['Train_Loss'] = tf.summary.scalar(
            'Train_Loss', tf.log(self.Output['Loss'])
        )
        self.Summary['Val_Loss'] = tf.summary.scalar(
            'Val_Loss', tf.log(self.Output['Loss'])
        )

        self.Summary['Weight'] = {}
        for key in ['Snip', 'Random']:
            self.Summary['Weight'][key] = [
                tf.summary.histogram(weight.name,
                    tf.boolean_mask(weight, tf.not_equal(weight, ZERO_32)))
                for weight in self.Model[key].Snip['Comb']
            ]

        self.Output['Pred'] = {
            'Snip': self.Output['Snip_Pred'],
            'Random': self.Output['Random_Pred'],
        }

        self.train_summary = [
            self.Summary['Train_Error'],
            self.Summary['Train_Loss']
        ]
        self.val_summary = [
            self.Summary['Val_Error'],
            self.Summary['Val_Loss']
        ]
        self.train_op = [
            self.Output['Snip_Train'],
            self.Output['Random_Train'],
        ]

    def preprocess(self, features, labels):
        # self.Dataset.train.next_batch(self.params.batch_size)

        feed_dict = {
            self.Placeholder['Input_Feature']: features,
            self.Placeholder['Input_Label']: labels,
        }
        self.Sess.run(tf.global_variables_initializer(), feed_dict=feed_dict)
        self.Mask['Random'] = {}
        self.Mask['Delta'] = {}

        random_masks = self.Sess.run(self.Model['Random'].Snip['Weight'])
        delta_masks = self.Sess.run(self.Model['Snip'].Snip['Weight'])
        for ix, (mask, delta) in enumerate(zip(random_masks, delta_masks)):
            binary = self._npr.binomial(1, 1 - self.params.random_k, mask.shape)
            self.Mask['Random'][self.Model['Random'].Snip['Mask'][ix]] = binary

            perturb = np.ones(delta.shape)

            self.Mask['Delta'][self.Model['Snip'].Snip['Mask'][ix]] = \
                perturb

        self.Mask['Index'] = self.Sess.run(self.Tensor['Snip_Grad'],
           {**self.Mask['Random'], **self.Mask['Delta'], **feed_dict}
        )

        self.Mask['Snip'] = {}

        if self.params.prune_method == 'together':
            self.prune_together()

        elif self.params.prune_method == 'separate' or 'weighted':
            self.prune_separate()

    def train(self, i, features, labels):
        # self.Dataset.train.next_batch(self.params.batch_size)
        # print(features, labels)

        feed_dict = {
            **self.Mask['Random'],
            **self.Mask['Snip'],
            self.Placeholder['Input_Feature']: features,
            self.Placeholder['Input_Label']: labels,
            self.Placeholder['Learning_Rate']: self.learning_rate
        }
        pred, *_ = self.Sess.run(
            [self.Output['Pred']] + self.train_op,
            feed_dict
        )
        for key in pred:
            summary = self.Sess.run(
                self.train_summary + self.Summary['Weight'][key],
                {**feed_dict, self.Placeholder['Input_Logits']: pred[key]}
            )

            for summ in summary:
                self.Writer[key].add_summary(summ, i)

        self.learning_rate = self.decay_lr(i, self.learning_rate)
        return features, labels

    def val(self, i):
        features, labels = self.Data[1]
        # self.Dataset.test.images, self.Dataset.test.labels

        feed_dict = {
            **self.Mask['Random'],
            **self.Mask['Snip'],
            self.Placeholder['Input_Feature']: features,
            self.Placeholder['Input_Label']: labels,
        }
        pred = self.Sess.run(
            [self.Output['Pred']], feed_dict)

        pred = pred[0]
        for key in pred:
            summary = self.Sess.run(
                self.val_summary,
                {**feed_dict, self.Placeholder['Input_Logits']: pred[key]}
            )
            for summ in summary:
                self.Writer[key].add_summary(summ, i)

    def run(self):
        features, labels = self._get_batch()
        self.preprocess(features, labels)

        for e in range(self.params.num_steps):
            features, labels = self._get_batch()
            print(features.shape)
            self.train(e, features, labels)
            if e % self.params.val_steps == 0:
                self.val(e)

    def decay_lr(self, i, learning_rate):
        if self.params.decay_scheme == 'exponential':
            if (i+1) % self.params.decay_iter == 0:
                learning_rate *= self.params.decay_rate

        elif self.params.decay_scheme == 'none':
            pass

        return learning_rate

    def _get_batch(self):
        end_ix = self.start_ix + self.params.batch_size
        if end_ix > len(self.Data[0][0]):
            end_ix = end_ix - len(self.Data[0][0])

            features = np.concatenate(
                [self.Data[0][0][self.start_ix:],
                self.Data[0][0][:end_ix]],
                axis=0
            )
            labels = np.concatenate(
                [self.Data[0][1][self.start_ix:],
                self.Data[0][1][:end_ix]],
                axis=0
            )
        else:
            features = self.Data[0][0][self.start_ix:end_ix]
            labels = self.Data[0][1][self.start_ix:end_ix]
        self.start_ix = end_ix
        return features, labels

    def prune_together(self):
        total_shape, total_size, flat_grad = [], [], []
        for ix in range(len(self.Mask['Index'])):
            grad = self.Mask['Index'][ix]
            total_shape.append(grad.shape)
            total_size.append(grad.size)
            flat_grad.append(grad.flatten())

        flat_size = 1
        for size in total_size:
            flat_size += size

        flat_grad = np.concatenate(flat_grad, axis=0)

        k = int((1 - self.params.prune_k) * flat_size)

        zeros = np.zeros_like(flat_grad)

        if self.params.value_method == 'largest':
            ind = np.argpartition(np.abs(flat_grad), -k, axis=None)[-k:]
        elif self.params.value_method == 'smallest':
            ind = np.argpartition(-np.abs(flat_grad), -k, axis=None)[-k:]
        elif self.params.value_method == 'mixed':
            l = int(k / 2)
            ind = np.concatenate(
                [np.argpartition(np.abs(flat_grad), -k, axis=None)[-l:],
                 np.argpartition(np.abs(flat_grad), -k, axis=None)[:k - l]]
            )
        elif self.params.value_method == 'weighted':
            raise ValueError('Cannot prune together with weighted')

        zeros[ind] = 1

        start = 0

        for ix, (shape, size) in enumerate(zip(total_shape, total_size)):
            print(ix)
            end = start + size
            self.Mask['Snip'][self.Model['Snip'].Snip['Mask'][ix]] = \
                np.reshape(zeros[start:end], shape)
            start = end

    def prune_separate(self):
        for ix in range(len(self.Mask['Index'])):
            grad = self.Mask['Index'][ix]
            k = int((1 - self.params.prune_k) * grad.size)
            zeros = np.zeros_like(grad)

            if self.params.prune_method == 'weighted':
                k = (1 - self.params.prune_k) * grad.size
                k *= (ix / len(self.Mask['Index']) + 0.5)
                k = int(k)

            if self.params.value_method == 'largest':
                ind = np.argpartition(np.abs(grad), -k, axis=None)[-k:]
            elif self.params.value_method == 'smallest':
                ind = np.argpartition(-np.abs(grad), -k, axis=None)[-k:]
            elif self.params.value_method == 'mixed':
                l = int(k / 2)
                ind = np.concatenate(
                    [np.argpartition(np.abs(grad), -k, axis=None)[-l:],
                     np.argpartition(np.abs(grad), -k, axis=None)[:k - l]]
                )
            ind = np.unravel_index(ind, zeros.shape)

            self.Mask['Snip'][self.Model['Snip'].Snip['Mask'][ix]] = zeros