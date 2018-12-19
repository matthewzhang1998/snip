import tensorflow as tf
import numpy as np

from runner.base_runner import *
from runner.mnist2 import get_mnist_dataset
from model.mask import *
from model.lisp import *
from util.optimizer_util import *

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow import keras

# OPTIMIZING THE SNIP OBJECTIVE DIRECTLY

ZERO_32 = tf.constant(0, dtype=tf.float32)

class MNISTRunner(BaseRunner):
    def __init__(self, scope, params):
        super(MNISTRunner, self).__init__(scope, params)

        self.Mask = {}

        self._build_outputs()
        self._build_lisp()
        self._build_summary()
        self.Writer = {}

        self.Saver = tf.train.Saver()

        self._npr = np.random.RandomState(params.seed)

        self.learning_rate = params.learning_rate
        self.pretrain_learning_rate = params.pretrain_learning_rate

        self.Mask['Lisp'] = {}

        self.Writer['Random'] = \
            tf.summary.FileWriter(self.Dir+'/random', self.Sess.graph)
        self.Writer['Snip'] = \
            tf.summary.FileWriter(self.Dir+'/snip', self.Sess.graph)
        self.Writer['Lisp'] = \
            tf.summary.FileWriter(self.Dir+'/lisp', self.Sess.graph)
        self.Writer['Random_No_Train'] = \
            tf.summary.FileWriter(self.Dir+'/random_no_train', self.Sess.graph)

    def _build_lisp(self):
        with tf.variable_scope(self.scope):
            self.Model['Lisp'] = Lisp('lisp', self.params, 784, 10)
            self.Model['Random_No_Train'] = MaskCopy('random_no_train',
                self.params, 784, 10, target=self.Model['Random']
            )

        self.Placeholder['Proxy'] = self.Model['Lisp'].Tensor['Proxy']
        self.Placeholder['Lisp_Mask'] = self.Model['Lisp'].Snip['Mask']
        self.Placeholder['Predicted'] = self.Model['Lisp'].Tensor['Predicted']

        self.Tensor['Layer_Loss_Function'] = MSELoss
        self.Tensor['Lisp_Grad'] = self.Model['Lisp'].prune_backward(
            self.Placeholder['Input_Label'], self.Tensor['Loss_Function'],
            self.Tensor['Layer_Loss_Function']
        )

        self.Tensor['Random_No_Train_Mask'] = self.Model['Random_No_Train'].Placeholder

        self.Output['Random_No_Train_Pred'] = self.Model['Random_No_Train'].run(
            self.Placeholder['Input_Feature']
        )

        self.Output['Random_No_Train_Loss'] = tf.reduce_mean(
            self.Tensor['Loss_Function'](
                self.Output['Random_No_Train_Pred'], self.Placeholder['Input_Label']
            )
        )
        self.Output['Random_No_Train_Train'] = \
            self.Output['Optimizer'].minimize(self.Output['Random_No_Train_Loss'])

        self.Output['Lisp_Pred'] = \
            self.Model['Lisp'].run(self.Placeholder['Input_Feature'])

        self.Output['Lisp_Loss'] = tf.reduce_mean(
            self.Tensor['Loss_Function'](
                self.Output['Lisp_Pred'], self.Placeholder['Input_Label']
            )
        )
        self.Output['Lisp_Loss'] += self.params.weight_decay * \
            tf.reduce_sum(
                [tf.nn.l2_loss(t) for t in self.Model['Lisp'].Snip['Weight']]
            )

        self.Output['Lisp_Train'] = \
            self.Output['Optimizer'].minimize(self.Output['Lisp_Loss'])

        self.Model['Lisp'].prune(
            self.Tensor['Proto_Minibatch'], self.Tensor['Loss_Function']
        )
        self.Tensor['Lisp_True_Grad'] = self.Model['Lisp'].Tensor['Lisp_True_Grad']

        self.Tensor['Lisp_Meta'] = \
            [-tf.reduce_sum(tf.abs(grad)) for grad in self.Tensor['Lisp_True_Grad']]
        self.Output['Meta_Loss']['Lisp'] = tf.add_n(self.Tensor['Lisp_Meta'])
        self.Output['Renormalize'] = {}

        self.Tensor['Prior'] = {}
        self.Output['Store_Prior'] = {}

        for key in ['Random', 'Lisp']:
            self.Output['Meta_Loss'][key] /= \
                tf.cast(tf.reduce_sum(
                    [tf.reduce_prod(weight.shape) for weight in
                     self.Model[key].Snip['Weight']]), tf.float32
                )

            self.Tensor['Prior'][key] = [tf.Variable(tf.ones_like(weight),
                trainable=False) for weight in self.Model[key].Snip['Weight']]
            self.Output['Store_Prior'][key] = []
            for copy, weight in zip(self.Tensor['Prior'][key],
                self.Model[key].Snip['Weight']):
                self.Output['Store_Prior'][key].append(tf.assign(copy, weight))

                mu, sig = tf.nn.moments(weight, [0,1])
                mu_prior, sig_prior = tf.nn.moments(copy, [0,1])
                #self.Output['Meta_Loss'][key] += self.params.pretrain_kl_beta * \
                #    tf.log(sig_prior/sig) + 0.5*(sig**2+(mu-mu_prior)**2)/(sig_prior**2)

            #self.Output['Meta_Loss'][key] += self.params.pretrain_weight_decay * \
            #    tf.reduce_sum(
            #        [tf.nn.l2_loss(t) for t in self.Model[key].Snip['Weight']]
            #    )
            self.Output['Meta_Train'][key] = \
                self.Output['Optimizer'].minimize(
                    self.Output['Meta_Loss'][key],
                    var_list=self.Model[key].Snip['Weight']
                )
            self.Output['Renormalize'][key] = []
            for weight in self.Model[key].Snip['Weight']:
                #moments = tf.nn.moments(weight, axes=[0,1])
                #new_weight = (weight - moments[0])/moments[1] * \
                #    2/tf.sqrt(tf.cast(tf.reduce_prod(tf.shape(weight)), tf.float32))
                new_weight = weight/tf.reduce_sum(tf.abs(weight))
                self.Output['Renormalize'][key].append(
                    tf.assign(weight, new_weight)
                )

        self.Output['Pred'] = {
            'Random': self.Output['Random_Pred'],
            'Snip': self.Output['Snip_Pred'],
            'Lisp': self.Output['Lisp_Pred'],
            'Random_No_Train': self.Output['Random_No_Train_Pred']
        }

    def _build_outputs(self):
        with tf.variable_scope(self.scope):
            self.Model['Snip'] = Mask('snip', self.params, 784, 10)
            self.Model['Random'] = Mask('random', self.params, 784, 10)

        self.Dataset = get_mnist_dataset() #input_data.read_data_sets("./tmp/data", one_hot=True)
        self.start_ix = 0

        self.Placeholder['Input_Feature'] = tf.placeholder(
            tf.float32, [None, 784]
        )
        self.Placeholder['Learning_Rate'] = tf.placeholder(
            tf.float32, []
        )

        self.Placeholder['Input_Label'] = tf.placeholder(
            tf.float32, [None, 10]
        )

        self.Placeholder['Input_Logits'] = tf.placeholder(tf.float32, [None, 10])

        self.Placeholder['Random_Mask'] = self.Model['Random'].Placeholder
        self.Placeholder['Snip_Mask'] = self.Model['Snip'].Placeholder

        self.Tensor['Proto_Minibatch'] = {
            'Features': self.Placeholder['Input_Feature'],
            'Labels': self.Placeholder['Input_Label']
        }

        self.Tensor['Loss_Function'] = \
            SoftmaxCE

        self.Output['Optimizer'] = get_optimizer(
            self.params, self.Placeholder['Learning_Rate']
        )

        self.Tensor['Snip_Index'] = self.Model['Snip'].prune(
            self.Tensor['Proto_Minibatch'], self.Tensor['Loss_Function']
        )
        self.Model['Random'].prune(
            self.Tensor['Proto_Minibatch'], self.Tensor['Loss_Function']
        )

        self.Tensor['Snip_Grad'] = self.Model['Snip'].Tensor['Snip_Grad']
        self.Tensor['Random_Grad'] = self.Model['Random'].Tensor['Snip_Grad']

        self.Output['Meta_Loss'] = {}
        self.Output['Meta_Train'] = {}

        self.Tensor['Snip_Meta'] = \
            [-tf.reduce_sum(tf.abs(grad)) for grad in self.Tensor['Snip_Grad']]
        self.Output['Meta_Loss']['Snip'] = tf.add_n(self.Tensor['Snip_Meta'])

        self.Tensor['Random_Meta'] = \
            [-tf.reduce_sum(tf.abs(grad)) for grad in self.Tensor['Random_Grad']]
        self.Output['Meta_Loss']['Random'] = tf.add_n(self.Tensor['Random_Meta'])

        self.Output['Random_Pred'] = self.Model['Random'].run(
            self.Placeholder['Input_Feature']
        )
        self.Output['Snip_Pred'] = self.Model['Snip'].run(
            self.Placeholder['Input_Feature']
        )
        self.Output['Pred'] = {
            'Snip': self.Output['Snip_Pred'],
            'Random': self.Output['Random_Pred']
        }

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

        self.Output['Random_Round'] = \
            tf.argmax(self.Output['Random_Pred'], 1)
        self.Output['Snip_Round'] = \
            tf.argmax(self.Output['Snip_Pred'], 1)

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

        self.Summary['Meta_Loss'] = {key: tf.summary.scalar(
            'Meta_Loss', self.Output['Meta_Loss'][key]
        ) for key in self.Output['Meta_Loss']}

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
        for key in ['Snip', 'Random', 'Random_No_Train', 'Lisp']:
            self.Summary['Weight'][key] = [
                tf.summary.histogram(weight.name,
                tf.boolean_mask(weight, tf.not_equal(weight, ZERO_32)))
                for weight in self.Model[key].Snip['Comb']
            ]

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
            self.Output['Lisp_Train'],
            self.Output['Random_No_Train_Train']
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
            binary = self._npr.binomial(1, 1-self.params.random_k, mask.shape)
            self.Mask['Random'][self.Model['Random'].Snip['Mask'][ix]] = \
                binary

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

        self.preprocess_lisp(features, labels)

        self.Mask['Random_No_Train'] = {}
        self.Sess.run(self.Output['Store_Prior'])
        for k in range(len(self.Mask['Random'])):
            self.Mask['Random_No_Train'][self.Model['Random_No_Train'].Snip['Mask'][k]] = \
                self.Mask['Random'][self.Model['Random'].Snip['Mask'][k]]

        feed_dict = {
            **self.Mask['Snip'],
            **self.Mask['Random'],
            **self.Mask['Lisp'],
            **self.Mask['Random_No_Train']
        }

        for e in range(self.params.pretrain_num_steps):
            self.pretrain(e)

        weights = self.Sess.run(self.Summary['Weight'], feed_dict)
        for key in weights:
            for weight in weights[key]:
                self.Writer[key].add_summary(weight)

    def pretrain(self, i):
        features, labels = self._get_batch()
        feed_dict = {
            **self.Mask['Snip'],
            **self.Mask['Random'],
            **self.Mask['Lisp'],
            **self.Mask['Random_No_Train'],
            self.Placeholder['Input_Feature']: features,
            self.Placeholder['Input_Label']: labels,
            self.Placeholder['Learning_Rate']: self.pretrain_learning_rate
        }

        summary, _ = self.Sess.run(
            [self.Summary['Meta_Loss'], self.Output['Meta_Train']],
            feed_dict
        )

        for key in summary:
            self.Writer[key].add_summary(summary[key], i)
        self.pretrain_learning_rate = self.decay_lr(i, self.pretrain_learning_rate)

    def train(self, i, features, labels):
        # self.Dataset.train.next_batch(self.params.batch_size)
        #print(features, labels)

        feed_dict = {
            **self.Mask['Random'],
            **self.Mask['Snip'],
            **self.Mask['Lisp'],
            **self.Mask['Random_No_Train'],
            self.Placeholder['Input_Feature']: features,
            self.Placeholder['Input_Label']: labels,
            self.Placeholder['Learning_Rate']: self.learning_rate
        }
        pred, *_ = self.Sess.run(
            [self.Output['Pred']]+self.train_op,
            feed_dict
        )
        for key in pred:
            summary = self.Sess.run(
                self.train_summary,
                {**feed_dict, self.Placeholder['Input_Logits']: pred[key]}
            )
            for summ in summary:
                self.Writer[key].add_summary(summ, i)

        self.learning_rate = self.decay_lr(i, self.learning_rate)
        return features, labels

    def preprocess_lisp(self, features, labels):
        delta_masks = self.Sess.run(self.Model['Lisp'].Snip['Weight'])

        for ix in reversed(range(len(delta_masks))):
            weight = delta_masks[ix]
            dummy = np.ones_like(weight)

            if ix == len(delta_masks) - 1:
                pred = labels
            else:
                pred = sample

            if ix != 0:
                sample = self._npr.normal(loc=0.0, scale=1.0,
                    size=[self.Dataset[0][0].shape[0], weight.shape[0]])
            else:
                sample = features

            feed_dict = {
                self.Placeholder['Proxy'][ix]: sample,
                self.Placeholder['Predicted'][ix]: pred,
                self.Placeholder['Lisp_Mask'][ix]: dummy,
                self.Placeholder['Input_Label']: labels,
                **self.Mask['Lisp']
            }
            grad = self.Sess.run(self.Tensor['Lisp_Grad'][ix], feed_dict)[0]
            mask = self.prune_layer(grad)
            self.Mask['Lisp'][self.Placeholder['Lisp_Mask'][ix]] = mask

    def val(self, i):
        features, labels = self.Dataset[1]
        # self.Dataset.test.images, self.Dataset.test.labels

        feed_dict = {
            **self.Mask['Random'],
            **self.Mask['Snip'],
            **self.Mask['Lisp'],
            **self.Mask['Random_No_Train'],
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
        features, labels = self.Dataset[0]
        self.preprocess(features, labels)

        for e in range(self.params.num_steps):
            features, labels = self._get_batch()
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
        if end_ix > len(self.Dataset[0][0]):
            end_ix = end_ix - len(self.Dataset[0][0])

            features = np.concatenate(
                [self.Dataset[0][0][self.start_ix:],
                self.Dataset[0][0][end_ix:]],
                axis=0
            )
            labels = np.concatenate(
                [self.Dataset[0][1][self.start_ix:],
                self.Dataset[0][1][end_ix:]],
                axis=0
            )
        else:
            features = self.Dataset[0][0][self.start_ix:end_ix]
            labels = self.Dataset[0][1][self.start_ix:end_ix]
        self.start_ix = end_ix
        return features, labels

    def prune_layer(self, grad):
        k = int((1 - self.params.prune_k) * grad.size)
        zeros = np.zeros_like(grad)

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
        zeros[ind] = 1
        return zeros

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
            zeros[ind] = 1

            self.Mask['Snip'][self.Model['Snip'].Snip['Mask'][ix]] = zeros

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

        # self.Mask['Snip'][self.Model['Snip'].Placeholder[-1]] = \
        #    np.ones_like(self.Mask['Index'][-1])