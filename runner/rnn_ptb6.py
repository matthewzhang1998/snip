import os.path as osp

import numpy as np
import tensorflow as tf
from model.mask import *
from runner.base_runner import *
from util.optimizer_util import *
from model.unit3 import *
from util.logger_util import *
from util.sparse_util import *
import scipy.misc
from collections import defaultdict

from tensorflow.contrib import slim

from data.load_pen import *

ZERO_32 = tf.constant(0.0, dtype=tf.float32)

class PTBRunner(BaseRunner):
    def __init__(self, scope, params):
        super(PTBRunner, self).__init__(scope, params)

        self._npr = np.random.RandomState(params.seed)
        self.Mask = {}
        self.Data = Dataset(self.params, "../data/simple-examples/data")
        self.vocab_size = self.Data.vocab_size
        self._build_snip()

        self._preprocess()
        self.Writer = {}

        self.Saver = tf.train.Saver()

        self.learning_rate = params.learning_rate
        self.pretrain_learning_rate = params.pretrain_learning_rate

        self.Writer['Unit'] = \
            FileWriter(self.Dir+'/unit', self.Sess.graph)

    def _build_snip(self):
        with tf.variable_scope(self.scope):
            self.Model['Unit'] = Unit('unit', self.params,
                self.vocab_size, self.vocab_size, self.params.seed)

            self.start_ix = 0

            self.Placeholder['Input_Feature'] = tf.placeholder(
                shape=[None, None], dtype=tf.int32,
            )

            self.Placeholder['Learning_Rate'] = tf.placeholder(
                tf.float32, []
            )

            self.Placeholder['Input_Label'] = tf.placeholder(
                tf.int32, [None, None]
            )

            self.Placeholder['Input_Logits'] = tf.placeholder(tf.float32,
                [None, None, self.vocab_size])

            self.Tensor['Proto_Minibatch'] = {
                'Features': self.Placeholder['Input_Feature'],
                'Labels': self.Placeholder['Input_Label']
            }

            self.Tensor['Loss_Function'] = \
                Seq2SeqLoss

            self.Output['Optimizer'] = get_optimizer(
                self.params, self.Placeholder['Learning_Rate']
            )

            self.Model['Unit'].unit(
                self.Tensor['Proto_Minibatch'], self.Tensor['Loss_Function'],
                len(self.Model['Unit'].Snip['Dummy_Kernel']) - 1
            )

            self.Tensor['Unit_Grad'] = self.Model['Unit'].Tensor['Unit_Grad']

            self.Placeholder['Unit_Kernel'] = self.Model['Unit'].Snip['Dummy_Kernel']

            self.Tensor['Variable_Initializer'] = {}

    def _preprocess(self):
        self.Sess.run(tf.global_variables_initializer())

        for i in reversed(range(len(self.Model['Unit'].Snip['Dummy_Kernel']))):
            self._preprocess_unit(i)

        self._build_summary()
        self.Sess.run(tf.variables_initializer(self.Output['Optimizer'].variables()))

    def _preprocess_unit(self, i):
        info = self.Model['Unit'].Info['Params'][i]
        type = self.Model['Unit'].Info['Type'][i]

        final_list = []

        features, labels = self._get_batch()
        if type == 'rnn':
            use_dense = False

            if 'lstm' in info['recurrent_cell_type']:
                nh = info['hidden_size']
                ni = info['input_depth']
                nu = self.params.num_unitwise_rnn
                nu = nh if nu > nh else nu

                h_ix = int((1-self.params.unit_k)*(ni+nh)*4*nh/(nh//nu+1))
                t_ix = h_ix*(nh//nu+1)
                top_vals = np.zeros((t_ix, 3), dtype=np.float32)
                rand_vals = np.zeros((t_ix, 3), dtype=np.float32)

                ix = 0

                def he_initializer(shape, npr, stddev=1.0):
                    out = npr.normal(loc=0, scale=shape[0], size=shape).astype(np.float32)
                    return out

                for j in range(nh//nu+1):
                    weights = he_initializer((ni+nh,4*nu), self._npr)

                    feed_dict = {
                        self.Placeholder['Unit_Kernel'][i]: weights,
                        self.Placeholder['Input_Feature']: features,
                        self.Placeholder['Input_Label']: labels,
                    }
                    grads, pred = self.Sess.run(
                        [self.Tensor['Unit_Grad'][i], self.Model['Unit'].Tensor['Unit_Pred']], feed_dict
                    )

                    grads = grads[0]

                    scipy.misc.imsave(osp.join(self.Dir, 'grad{}.jpg'.format(info['scope'])), grads)
                    top_k = np.unravel_index(
                        np.argpartition(np.abs(grads), -h_ix, axis=None)[-h_ix:],
                        (ni+nh,4*nu)
                    )
                    random_k = np.unravel_index(self._npr.choice(np.arange(weights.size),
                        size = (h_ix,), replace=False),
                        (ni+nh,4*nu))

                    for k in range(len(top_k[0])):
                        l,m = top_k[0][k], top_k[1][k]

                        l2, m2 = random_k[0][k], random_k[1][k]
                        if j*nu + m%nu >= nh:
                            # ignore
                            top_vals[ix] = [0,0,0]

                        else:
                            top_vals[ix] = [weights[l][m], l, m%nu + j*nu + m//nu*nh]

                        if j*nu + m2%nu >= nh:
                            # ignore
                            rand_vals[ix] = [0,0,0]

                        else:
                            rand_vals[ix] = [weights[l2][m2], l2, m2%nu + j*nu + m2//nu*nh]

                        ix += 1

                random_list = [rand_vals[:,0], rand_vals[:, 1:]]
                top_list = [top_vals[:,0], top_vals[:, 1:]]

        elif type == 'mlp' and info['scope'] != 'softmax':
            use_dense = True
            nh = info['hidden_size']
            ni = info['input_depth']

            if info['scope'] == 'softmax':
                k_ratio = self.params.softmax_sparsity
            elif 'mlp' in info['scope']:
                k_ratio = self.params.mlp_sparsity
            else:
                k_ratio = 0.99

            t_ix = int(nh*ni*(1-k_ratio))
            top_vals = np.zeros((t_ix, 3))
            rand_vals = np.zeros((t_ix, 3))

            weights = np.load(osp.join('../weights/rnn', '{}.npy'.format(i)))
            random_list = top_list = weights
            all_weights = weights

            scipy.misc.imsave(osp.join(self.Dir, '{}.jpg'.format(info['scope'])), weights)

        elif type == 'embedding' or info['scope'] == 'softmax':
            use_dense = True
            nh = info['hidden_size']
            ni = info['input_depth']

            if info['scope'] == 'softmax':
                k_ratio = self.params.softmax_sparsity
            elif info['scope'] == 'embed':
                k_ratio = self.params.embed_sparsity

            t_ix = int(nh * ni * (1 - k_ratio))

            weights = np.load(osp.join('../weights/rnn', '{}.npy'.format(i)))
            random_list = top_list = weights
            all_weights = weights

            scipy.misc.imsave(osp.join(self.Dir, '{}.jpg'.format(info['scope'])), weights)

        self._build_networks(top_list, random_list, i, use_dense=use_dense)

        np.save(osp.join(self.Dir, '{}'.format(i)), all_weights)
        self.Sess.run(self.Tensor['Variable_Initializer'])

    def _build_networks(self, unit_list, random_list, i, use_dense=False):
        self.Model['Unit'].build_sparse(unit_list, i, use_dense=use_dense)

        if i != 0:
            self.Model['Unit'].unit(
                self.Tensor['Proto_Minibatch'], self.Tensor['Loss_Function'],
                i - 1
            )

            self.Tensor['Unit_Grad'] = self.Model['Unit'].Tensor['Unit_Grad']

            self.Placeholder['Unit_Kernel'] = self.Model['Unit'].Snip['Dummy_Kernel']

            for key in self.Model:
                self.Tensor['Variable_Initializer'][key] = self.Model[key].initialize_op

        else:

            for key in self.Model:
                self.Tensor['Variable_Initializer'][key] = self.Model[key].initialize_op

            self.Output['Unit_Pred'] = self.Model['Unit'].run(
                self.Placeholder['Input_Feature']
            )

            self.Output['Unit_Loss'] = tf.reduce_mean(
                self.Tensor['Loss_Function'](
                    self.Output['Unit_Pred'], self.Placeholder['Input_Label']
                )
            )
            self.Output['Unit_Train'] = \
                self.Output['Optimizer'].minimize(self.Output['Unit_Loss'])

    def _build_summary(self):
        self.Output['Loss'] = tf.reduce_mean(
            self.Tensor['Loss_Function'](
                self.Placeholder['Input_Logits'],
                self.Placeholder['Input_Label']
            )
        )

        self.Placeholder['Val_Error'] = tf.placeholder(
            dtype=tf.float32, shape=[]
        )
        self.Placeholder['Val_Loss'] = tf.placeholder(
            dtype=tf.float32, shape=[]
        )

        self.Output['Error'] = tf.exp(self.Output['Loss'])

        self.val_res = {
            'Val_Error': self.Output['Error'],
            'Val_Loss': self.Output['Loss']
        }

        self.val_placeholder = {
            'Val_Error': self.Placeholder['Val_Error'],
            'Val_Loss': self.Placeholder['Val_Loss']
        }

        self.Summary['Train_Error'] = tf.summary.scalar(
            'Train_Error', self.Output['Error']
        )
        self.Summary['Val_Error'] = tf.summary.scalar(
            'Val_Error', self.Placeholder['Val_Error']
        )

        self.Summary['Train_Loss'] = tf.summary.scalar(
            'Train_Loss', tf.log(self.Output['Loss'])
        )
        self.Summary['Val_Loss'] = tf.summary.scalar(
            'Val_Loss', tf.log(self.Placeholder['Val_Loss'])
        )

        #self.Summary['Weight'] = {}
        #for key in ['Random', 'Unit']:
        #    self.Summary['Weight'][key] = [
        #        tf.summary.histogram(weight.name,
        #            tf.boolean_mask(weight, tf.not_equal(weight, ZERO_32)))
        #        for weight in self.Model.Sparse
        #    ]

        self.Output['Pred'] = {
            'Unit': self.Output['Unit_Pred']
        }

        self.train_summary = {
            'Train_Error': self.Output['Error'],
            'Train_Loss': self.Output['Loss']
        }
        self.val_summary = {
            'Val_Error': self.Placeholder['Val_Error'],
            'Val_Loss': self.Placeholder['Val_Loss']
        }
        self.train_op = [
            self.Output['Unit_Train']
        ]

    def train(self, i, features, labels):
        # self.Dataset.train.next_batch(self.params.batch_size)
        # print(features, labels)

        print(i)

        feed_dict = {
            self.Placeholder['Input_Feature']: features,
            self.Placeholder['Input_Label']: labels,
            self.Placeholder['Learning_Rate']: self.learning_rate
        }
        pred, *_ = self.Sess.run(
            [self.Output['Pred']] + self.train_op,
            feed_dict
        )
        #self.Writer['Unit'].add_run_metadata(self.Sess.rmd, 'train' + str(i))
        for key in pred:
            summary = self.Sess.run(
                self.train_summary,
                {**feed_dict, self.Placeholder['Input_Logits']: pred[key]}
            )

            self.Writer[key].add_summary(summary, i)

        self.learning_rate = self.decay_lr(i, self.learning_rate)
        return features, labels

    def val(self, i):
        start = 0
        summary = {'Unit': defaultdict(list)}

        for k in range(self.params.val_size):
            end = start + self.params.batch_size

            b_feat, b_lab = self._get_batch('val')
            # self.Dataset.test.images, self.Dataset.test.labels

            feed_dict = {
                self.Placeholder['Input_Feature']: b_feat,
                self.Placeholder['Input_Label']: b_lab,
            }
            pred = self.Sess.run(
                [self.Output['Pred']], feed_dict)

            pred = pred[0]
            for key in pred:
                b_summary = self.Sess.run(
                    self.val_res,
                    {**feed_dict, self.Placeholder['Input_Logits']: pred[key]}
                )

                for summ in b_summary:
                    summary[key][summ].append(b_summary[summ])

        for key in summary:
            for summ in summary[key]:
                summary[key][summ] = np.mean(summary[key][summ])

            write_summary = self.Sess.run(
                self.val_summary,
                {self.val_placeholder[summ]: summary[key][summ]
                 for summ in summary[key]}
            )
            self.Writer[key].add_summary(write_summary, i)

    def run(self):
        #slim.model_analyzer.analyze_vars(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES), print_info=True)

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

    def _get_batch(self, type='train'):
        return self.Data.get_batch(type)