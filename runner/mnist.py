import tensorflow as tf

from runner.base_runner import *
from util.optimizer_util import *

from tensorflow.examples.tutorials.mnist import input_data

def SoftmaxCE(logits, labels):
    return tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=labels
    )

class MNISTRunner(BaseRunner):
    def __init__(self, scope, params):
        super(MNISTRunner, self).__init__(scope, params)

        self._build_outputs()
        self._build_summary()
        self.Saver = tf.train.Saver()

    def _build_outputs(self):
        with tf.variable_scope(self.scope):
            self.Model['Snip'] = Snip('snip', self.params, 784, 10)
            self.Model['Random'] = Random('random', self.params, 784, 10)

        self.Dataset = input_data.read_data_sets("/tmp/data", one_hot=True)

        self.Placeholder['Input_Feature'] = tf.placeholder(
            tf.float32, [None, 784]
        )
        self.Placeholder['Input_Label'] = tf.placeholder(
            tf.float32, [None, 10]
        )

        self.Tensor['Proto_Minibatch'] = {
            'Features': self.Placeholder['Input_Feature'],
            'Labels': self.Placeholder['Input_Label']
        }

        self.Tensor['Loss_Function'] = \
            SoftmaxCE

        self.Output['Optimizer'] = get_optimizer(self.params)

        self.Model['Snip'].prune(
            self.Tensor['Proto_Minibatch'], self.Tensor['Loss_Function']
        )

        self.Output['Random_Pred'] = self.Model['Random'].run(
            self.Placeholder['Input_Feature']
        )

        self.Output['Snip_Pred'] = self.Model['Snip'].run(
            self.Placeholder['Input_Feature']
        )

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

        self.Output['Random_Round'] = \
            tf.round(tf.sigmoid(self.Output['Random_Pred']))

        self.Output['Snip_Round'] = \
            tf.round(tf.sigmoid(self.Output['Snip_Pred']))

        self.Output['Random_Accuracy'] = tf.reduce_mean(
            tf.cast(tf.equal(
                self.Output['Random_Round'],
                self.Placeholder['Input_Label']
            ), tf.float32)
        )

        self.Output['Snip_Accuracy'] = tf.reduce_mean(
            tf.cast(tf.equal(
                self.Output['Snip_Round'],
                self.Placeholder['Input_Label']
            ), tf.float32)
        )

        self.Output['Random_Train'] = \
            self.Output['Optimizer'].minimize(self.Output['Random_Loss'])
        self.Output['Snip_Train'] = \
            self.Output['Optimizer'].minimize(self.Output['Snip_Loss'])


    def _build_summary(self):
        self.Summary['Snip_Train_Accuracy'] = tf.summary.scalar(
            'Snip_Train_Accuracy', self.Output['Snip_Accuracy']
        )
        self.Summary['Snip_Val_Accuracy'] = tf.summary.scalar(
            'Snip_Val_Accuracy', self.Output['Snip_Accuracy']
        )

        self.Summary['Snip_Train_Loss'] = tf.summary.scalar(
            'Snip_Train_Loss', self.Output['Snip_Loss']
        )
        self.Summary['Snip_Val_Loss'] = tf.summary.scalar(
            'Snip_Val_Loss', self.Output['Snip_Loss']
        )

        self.Summary['Random_Train_Accuracy'] = tf.summary.scalar(
            'Random_Train_Accuracy', self.Output['Random_Accuracy']
        )
        self.Summary['Random_Val_Accuracy'] = tf.summary.scalar(
            'Random_Val_Accuracy', self.Output['Random_Accuracy']
        )

        self.Summary['Random_Train_Loss'] = tf.summary.scalar(
            'Random_Train_Loss', self.Output['Random_Loss']
        )
        self.Summary['Random_Val_Loss'] = tf.summary.scalar(
            'Random_Val_Loss', self.Output['Random_Loss']
        )

        self.train_summary = [
            self.Summary['Random_Train_Accuracy'],
            self.Summary['Random_Train_Loss'],
            self.Summary['Snip_Train_Accuracy'],
            self.Summary['Snip_Train_Loss']
        ]

        self.val_summary = [
            self.Summary['Random_Val_Accuracy'],
            self.Summary['Random_Val_Loss'],
            self.Summary['Snip_Val_Accuracy'],
            self.Summary['Snip_Val_Loss']
        ]

        self.train_op = [
            self.Output['Random_Train'],
            self.Output['Snip_Train']
        ]

    def preprocess(self):
        features, labels = self.Dataset.train.next_batch(self.params.batch_size)

        feed_dict = {
            self.Placeholder['Input_Feature']: features,
            self.Placeholder['Input_Label']: labels,
        }
        self.Sess.run(tf.global_variables_initializer(), feed_dict=feed_dict)

        assign_op = self.Model['Snip'].Op['Delta']
        self.Sess.run(assign_op, feed_dict={})

        snip_op = self.Model['Snip'].Op['Sparse']
        self.Sess.run(snip_op, feed_dict)

        random_op = self.Model['Random'].Op['Sparse']
        self.Sess.run(random_op, feed_dict={})

    def train(self, i):
        features, labels = self.Dataset.train.next_batch(self.params.batch_size)

        feed_dict = {
            self.Placeholder['Input_Feature']: features,
            self.Placeholder['Input_Label']: labels,
        }

        summary = \
            self.Sess.run(self.train_summary + self.train_op, feed_dict)

        for summ in summary[:-2]:
            self.Writer.add_summary(summ, i)

    def val(self, i):
        features, labels = self.Dataset.test.images, self.Dataset.test.labels

        feed_dict = {
            self.Placeholder['Input_Feature']: features,
            self.Placeholder['Input_Label']: labels,
        }
        summary = self.Sess.run(self.val_summary, feed_dict)

        for summ in summary:
            self.Writer.add_summary(summ, i)

    def run(self):
        self.preprocess()

        for e in range(self.params.num_steps):
            self.train(e)
            if e % self.params.val_steps == 0:
                self.val(e)