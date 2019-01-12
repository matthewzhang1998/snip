import os.path as osp

import numpy as np
import tensorflow as tf
from model.vanilla import *
from runner.base_runner import *
from util.optimizer_util import *
from util.logger_util import *
from model.snip import *
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

        self._build_summary()
        self.Writer = {}

        self.Saver = tf.train.Saver()

        self.learning_rate = params.learning_rate
        self.pretrain_learning_rate = params.pretrain_learning_rate

        self.Writer['Small'] = \
            FileWriter(self.Dir+'/small', self.Sess.graph)

    def _build_snip(self):
        with tf.variable_scope(self.scope):
            self.Model['Small'] = Vanilla('small', self.params,
                self.vocab_size, self.vocab_size, init_path=self.params.weight_dir)

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

            self.Tensor['Loss_Function'] = \
                Seq2SeqLoss

            self.Output['Optimizer'] = get_optimizer(
                self.params, self.Placeholder['Learning_Rate']
            )

            self.Output['Small_Pred'] = self.Model['Small'].run(
                self.Placeholder['Input_Feature']
            )

            self.Output['Small_Loss'] = tf.reduce_mean(
                self.Tensor['Loss_Function'](
                    self.Output['Small_Pred'], self.Placeholder['Input_Label']
                )
            )
            self.Output['Small_Train'] = \
                self.Output['Optimizer'].minimize(self.Output['Small_Loss'])

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

        self.Output['Pred'] = {
            'Small': self.Output['Small_Pred']
        }

        self.train_summary = {
            'Train_Error': self.Output['Error'],
            'Train_Loss': tf.log(self.Output['Loss'])
        }
        self.val_summary = {
            'Val_Error': self.Placeholder['Val_Error'],
            'Val_Loss': tf.log(self.Placeholder['Val_Loss'])
        }
        self.train_op = [
            self.Output['Small_Train']
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
        summary = {'Small': defaultdict(list)}

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
        self.Sess.run(tf.global_variables_initializer())

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