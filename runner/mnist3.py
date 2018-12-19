import tensorflow as tf

from runner.base_runner import *
from model.lisp import *
from util.optimizer_util import *
from runner.mnist2 import *

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow import keras

''' LAYERWISE INITIALIZATION WITH SPARSITY '''

class MNIST3Runner(MNISTRunner):
    def __init__(self, scope, params):
        super(MNIST3Runner, self).__init__(scope, params)

        self.Mask['Lisp'] = {}
        self.Mask['Fore'] = {}
        self._build_lisp()

        self.Writer = {}
        # overwrite
        self.Writer['Random'] = \
            tf.summary.FileWriter(self.Dir+'/random', self.Sess.graph)
        self.Writer['Snip'] = \
            tf.summary.FileWriter(self.Dir+'/snip', self.Sess.graph)
        self.Writer['Lisp'] = \
            tf.summary.FileWriter(self.Dir+'/lisp', self.Sess.graph)
        self.Writer['Fore'] = \
            tf.summary.FileWriter(self.Dir+'/fore_lisp', self.Sess.graph)

    def _build_lisp(self):
        with tf.variable_scope(self.scope):
            self.Model['Lisp'] = Lisp('lisp', self.params, 784, 10)

        self.Placeholder['Proxy'] = self.Model['Lisp'].Tensor['Proxy']
        self.Placeholder['Lisp_Mask'] = self.Model['Lisp'].Snip['Mask']['Lisp']
        self.Placeholder['Fore_Mask'] = self.Model['Lisp'].Snip['Mask']['Fore']
        self.Placeholder['Predicted'] = self.Model['Lisp'].Tensor['Predicted']

        self.Tensor['Layer_Loss_Function'] = MSELoss
        self.Tensor['Lisp_Grad'] = self.Model['Lisp'].prune_backward(
            self.Placeholder['Input_Label'], self.Tensor['Loss_Function'],
            self.Tensor['Layer_Loss_Function']
        )

        self.Tensor['Fore_Grad'] = self.Model['Lisp'].prune_forward(
            self.Placeholder['Input_Feature'], self.Tensor['Layer_Loss_Function']
        )

        self.Output['Lisp_Pred'], self.Output['Fore_Pred'] = \
            self.Model['Lisp'].run(self.Placeholder['Input_Feature'])

        self.Output['Lisp_Loss'] = tf.reduce_mean(
            self.Tensor['Loss_Function'](
                self.Output['Lisp_Pred'], self.Placeholder['Input_Label']
            )
        )
        self.Output['Fore_Loss'] = tf.reduce_mean(
            self.Tensor['Loss_Function'](
                self.Output['Fore_Pred'], self.Placeholder['Input_Label']
            )
        )
        self.Output['Lisp_Loss'] += self.params.weight_decay * \
            tf.reduce_sum(
                [tf.nn.l2_loss(t) for t in self.Model['Lisp'].Snip['Weight']['Lisp']]
            )
        self.Output['Fore_Loss'] += self.params.weight_decay * \
            tf.reduce_sum(
                [tf.nn.l2_loss(t) for t in self.Model['Lisp'].Snip['Weight']['Fore']]
            )

        self.Output['Lisp_Train'] = \
            self.Output['Optimizer'].minimize(self.Output['Lisp_Loss'])
        self.Output['Fore_Train'] = \
            self.Output['Optimizer'].minimize(self.Output['Lisp_Loss'])

        self.train_op.append(self.Output['Lisp_Train'])
        #self.train_op.append(self.Output['Fore_Train'])

        self.Output['Pred'] = {
            'Random': self.Output['Random_Pred'],
            'Snip': self.Output['Snip_Pred'],
            'Lisp': self.Output['Lisp_Pred'],
            #'Fore': self.Output['Fore_Pred']
        }

    def preprocess(self, features, labels):
        super(MNIST3Runner, self).preprocess(features, labels)

        delta_masks = self.Sess.run(self.Model['Lisp'].Snip['Weight']['Lisp'])

        for ix in reversed(range(len(delta_masks))):
            weight = delta_masks[ix]
            dummy = np.ones_like(weight)

            if ix == len(delta_masks) - 1:
                pred = labels
            else:
                pred = sample

            if ix != 0:
                sample = self._npr.normal(loc=0.0, scale=1.0,
                    size=[self.params.batch_size, weight.shape[0]])
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

    def train(self, i, features, labels):
        feed_dict = {
            **self.Mask['Random'],
            **self.Mask['Snip'],
            **self.Mask['Lisp'],
            **self.Mask['Fore'],
            self.Placeholder['Input_Feature']: features,
            self.Placeholder['Input_Label']: labels,
            self.Placeholder['Learning_Rate']: self.learning_rate
        }
        outputs, *_ = self.Sess.run(
            [self.Output['Pred']]+self.train_op,
            feed_dict)

        for key in outputs:
            summary = self.Sess.run(
                self.train_summary,
                {**feed_dict, self.Placeholder['Input_Logits']: outputs[key]}
            )
            for summ in summary:
                self.Writer[key].add_summary(summ, i)

        self.decay_lr(i)
        return features, labels


    def val(self, i):
        features, labels = self.Dataset[1]
        # self.Dataset.test.images, self.Dataset.test.labels

        feed_dict = {
            **self.Mask['Random'],
            **self.Mask['Snip'],
            **self.Mask['Lisp'],
            **self.Mask['Fore'],
            self.Placeholder['Input_Feature']: features,
            self.Placeholder['Input_Label']: labels,
        }
        outputs = self.Sess.run(
            self.Output['Pred'],
            feed_dict)

        for key in outputs:
            summary = self.Sess.run(
                self.val_summary,
                {**feed_dict, self.Placeholder['Input_Logits']: outputs[key]}
            )
            for summ in summary:
                self.Writer[key].add_summary(summ, i)