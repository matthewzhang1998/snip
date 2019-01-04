import numpy as np
import tensorflow as tf
from model.mask import *
from runner.base_runner import *
from util.optimizer_util import *
from model.unit2 import *
from util.sparse_util import *
from collections import defaultdict

from tensorflow.contrib import slim

from data.load_pen import *

ZERO_32 = tf.constant(0.0, dtype=tf.float32)

class PTBRunner(BaseRunner):
    def __init__(self, scope, params):
        super(PTBRunner, self).__init__(scope, params)

        self._npr = np.random.RandomState(params.seed)
        self.Mask = {}
        self.Data, self.vocab_size = build_data("../data/simple-examples/data")
        print(self.vocab_size)

        self._build_snip()

        self._preprocess_unit()
        self.Writer = {}

        self.Saver = tf.train.Saver()

        self.learning_rate = params.learning_rate
        self.pretrain_learning_rate = params.pretrain_learning_rate

        self.Writer['Random'] = \
            tf.summary.FileWriter(self.Dir+'/random', self.Sess.graph)
        self.Writer['Unit'] = \
            tf.summary.FileWriter(self.Dir+'/unit', self.Sess.graph)

    def _build_snip(self):
        with tf.variable_scope(self.scope):
            self.Tensor['Embedding'] = tf.get_variable('embedding',
                [self.vocab_size, self.params.embed_size], dtype=tf.float32,
                initializer=tf.initializers.random_uniform(-.1, .1))

            #self.Model['Random'] = Unit('random', self.params,
            #    self.params.embed_size, self.params.embed_size, self.params.seed)
            self.Model['Unit'] = Unit('unit', self.params,
                self.params.embed_size, self.params.embed_size, self.params.seed)

            self.start_ix = 0

            self.Placeholder['Input_Feature'] = tf.placeholder(
                shape=[None, None], dtype=tf.int32,
            )

            self.Tensor['Input_Embed'] = tf.nn.embedding_lookup(
                self.Tensor['Embedding'], self.Placeholder['Input_Feature']
            )

            self.Tensor['SoftMax_W'] = tf.get_variable(
                "softmax_w", [self.params.embed_size, self.vocab_size],
                initializer=tf.initializers.random_uniform(-.1, .1), dtype=tf.float32,
            )

            self.Tensor['SoftMax_B'] = tf.get_variable(
                "softmax_b", [self.vocab_size],
                dtype=tf.float32
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
                'Features': self.Tensor['Input_Embed'],
                'Labels': self.Placeholder['Input_Label']
            }

            self.Tensor['Loss_Function'] = \
                Seq2SeqLoss

            self.Output['Optimizer'] = get_optimizer(
                self.params, self.Placeholder['Learning_Rate']
            )

            self.Model['Unit'].unit(
                self.Tensor['Proto_Minibatch'], self.Tensor['Loss_Function'],
                (self.Tensor['SoftMax_W'], self.Tensor['SoftMax_B'])
            )

            self.Placeholder['Unit_Kernel'] = self.Model['Unit'].Snip['Dummy_Kernel']
            self.Placeholder['Unit_Bias'] = self.Model['Unit'].Snip['Dummy_Bias']

            self.Tensor['Unit_Grad'] = self.Model['Unit'].Tensor['Unit_Grad']

    def _preprocess_unit(self):
        self.Sess.run(tf.global_variables_initializer())

        features, labels = self._get_batch()
        final_list = []
        random_list = []
        length = 0
        nh = self.params.rnn_r_hidden_seq[0]
        nu = self.params.num_unitwise

        h_ix = int((1-self.params.unit_k)*(self.params.embed_size+2*nh)*4*nu)
        t_ix = h_ix*(nh//nu+1)
        top_vals = np.zeros((t_ix, 3), dtype=np.float32)
        ix = 0

        def normc_initializer(shape, npr, stddev=1.0):
            out = npr.randn(*shape).astype(np.float32)
            out *= stddev / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
            return out

        for i in range(nh//nu+1):
            print(i, nh//nu)

            weights = normc_initializer((self.params.embed_size+2*nh,4*nu), self._npr)
            biases = np.zeros([nu*4])

            feed_dict = {
                self.Placeholder['Unit_Kernel'][0]: weights,
                self.Placeholder['Unit_Bias'][0]: biases,
                self.Placeholder['Input_Feature']: features,
                self.Placeholder['Input_Label']: labels,
            }
            grads = self.Sess.run(
                self.Tensor['Unit_Grad'], feed_dict
            )[0]

            top_k = np.unravel_index(
                np.argpartition(np.abs(grads), -h_ix, axis=None)[-h_ix:],
                (self.params.embed_size+2*nh,4*nu)
            )
            for j in range(len(top_k[0])):
                k,l = top_k[0][j], top_k[1][j]
                if i*nu + l%nu >= nh:
                    # just take some random weight

                    top_vals[ix] = [weights[k][l], k, l%nu + (i-1)*nu + l//nu*nh]

                else:
                    top_vals[ix] = [weights[k][l], k, l%nu + i*nu + l//nu*nh]
                ix += 1
                # grad_val = np.abs(grads[k][l])
                #
                # length = len(final_list)
                # if length < k_ix:
                #     for m in range(length+1):
                #         if m == length:
                #             final_list.append(((k, l % nu + l // nu * nh), weights[k][l], grad_val))
                #             break
                #
                #         elif grad_val < final_list[m][-1]:
                #             final_list = final_list[:m] + \
                #                 [((k, l % nu + l // nu * nh), weights[k][l], grad_val)] + \
                #                 final_list[m:]
                #             break
                #
                # else:
                #     for m in range(length):
                #         if grad_val < final_list[m][-1]:
                #             final_list = final_list[:m] + \
                #                  [((k, l % nu + l // nu * nh), weights[k][l], grad_val)] + \
                #                  final_list[m:]
                #             break
                #
                #         elif m == length - 1:
                #             final_list.append(((k, l % nu + l // nu * nh), weights[k][l], grad_val))
                #     final_list.pop(0)

        random_list = [self._npr.randn(t_ix,),
            np.vstack([self._npr.randint(self.params.embed_size+2*nh, size=(t_ix,)),
            self._npr.randint(4*nh, size=(t_ix,))]).T
        ]
        final_list = [top_vals[:,0], top_vals[:,1:]]

        self._build_networks([final_list], [random_list])
        self._build_summary()

    def _build_networks(self, unit_list, random_list):
        #self.Model['Random'].build_sparse(random_list)
        self.Model['Unit'].build_sparse(unit_list)

        # self.Output['Random_Embed'] = self.Model['Random'].run(
        #     self.Tensor['Input_Embed']
        # )

        self.Output['Unit_Embed'] = self.Model['Unit'].run(
            self.Tensor['Input_Embed']
        )

        # self.Output['Random_Pred'] = tf.einsum('ijk,kl->ijl',
        #     self.Output['Random_Embed'], self.Tensor['SoftMax_W']) + self.Tensor['SoftMax_B']

        self.Output['Unit_Pred'] = tf.einsum('ijk,kl->ijl',
            self.Output['Unit_Embed'], self.Tensor['SoftMax_W']) + self.Tensor['SoftMax_B']

        #self.Output['Random_Loss'] = tf.reduce_mean(
        #    self.Tensor['Loss_Function'](
        #        self.Output['Random_Pred'], self.Placeholder['Input_Label']
        #    )
        #)

        self.Output['Unit_Loss'] = tf.reduce_mean(
            self.Tensor['Loss_Function'](
                self.Output['Unit_Pred'], self.Placeholder['Input_Label']
            )
        )
        #self.Output['Random_Train'] = \
        #    self.Output['Optimizer'].minimize(self.Output['Random_Loss'])
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
            #'Random': self.Output['Random_Pred'],
            'Unit': self.Output['Unit_Pred']
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
            #self.Output['Random_Train'],
            self.Output['Unit_Train']
        ]

    def train(self, i, features, labels):
        # self.Dataset.train.next_batch(self.params.batch_size)
        # print(features, labels)

        feed_dict = {
            self.Placeholder['Input_Feature']: features,
            self.Placeholder['Input_Label']: labels,
            self.Placeholder['Learning_Rate']: self.learning_rate
        }
        pred, *_ = self.Sess.run(
            [self.Output['Pred']] + self.train_op,
            feed_dict
        )
        self.Writer['Unit'].add_run_metadata(self.Sess.rmd, 'train' + str(i))
        for key in pred:
            summary = self.Sess.run(
                self.train_summary,
                {**feed_dict, self.Placeholder['Input_Logits']: pred[key]}
            )

            for summ in summary:
                self.Writer[key].add_summary(summ, i)
                self.Writer[key].flush()

        self.learning_rate = self.decay_lr(i, self.learning_rate)
        return features, labels

    def val(self, i):
        features, labels = self.Data['val']

        ix = np.arange(len(features))
        self._npr.shuffle(ix)
        start = 0
        summary = {'Unit': defaultdict(list),} #'Random': defaultdict(list)}

        for k in range(self.params.val_size):
            end = start + self.params.batch_size

            b_feat = features[ix[start:end]]
            b_lab = labels[ix[start:end]]
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
            for summ in write_summary:
                self.Writer[key].add_summary(summ, i)

    def run(self):
        self.Sess.run([self.Model[key].initialize_op for key in self.Model]+
                      [tf.variables_initializer(self.Output['Optimizer'].variables())])

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

    def _get_batch(self):
        end_ix = self.start_ix + self.params.batch_size
        if end_ix > len(self.Data['train'][0]):
            end_ix = end_ix - len(self.Data['train'][0])

            features = np.concatenate(
                [self.Data['train'][0][self.start_ix:],
                 self.Data['train'][0][:end_ix]],
                axis=0
            )
            labels = np.concatenate(
                [self.Data['train'][1][self.start_ix:],
                 self.Data['train'][1][:end_ix]],
                axis=0
            )
        else:
            features = self.Data['train'][0][self.start_ix:end_ix]
            labels = self.Data['train'][1][self.start_ix:end_ix]
        self.start_ix = end_ix
        return features, labels
