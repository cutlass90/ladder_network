import time
import math

import tensorflow as tf
from  tensorflow.contrib.layers.python.layers.layers import batch_norm
from tqdm import tqdm

from my_layers import batch_norm as noised_batch_norm
from model_abstract import Model

class Ladder(Model):

    # --------------------------------------------------------------------------
    def __init__(self, input_dim, n_classes, do_train, scope):

        self.input_dim = input_dim
        self.n_classes = n_classes
        self.do_train = do_train

        self.summary = []
        self.mean, self.variance, self.z_noised, self.z_clear = [], [], [], []
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(1)
            with tf.variable_scope(scope):
                self.create_graph()
            if do_train:
                self.cost = self.create_cost_graph(self.labels, self.logits_noised)
                self.create_summary(self.labels, self.logits_clear)
                self.train = self.create_optimizer_graph(self.cost)
                self.train_writer, self.test_writer = self.create_summary_writers('summary/ladder')
                self.merged = tf.summary.merge(self.summary)

            self.sess = self.create_session()
            self.sess.run(tf.global_variables_initializer())
            self.stored_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
            self.saver = tf.train.Saver(self.stored_vars, max_to_keep=1000)


    # --------------------------------------------------------------------------
    def create_graph(self):
        print('Creat graph')

        self.inputs,\
        self.labels,\
        self.weight_decay,\
        self.learn_rate,\
        self.keep_prob,\
        self.is_training = self.input_graph() # inputs shape is # b*n_f x h1 x c1

        self.logits_clear = self.encoder(self.inputs,
            structure=[1000, 500, 250, self.n_classes], reuse=False, noise_std=None)
        self.logits_noised = self.encoder(self.inputs,
            structure=[1000, 500, 250, self.n_classes], reuse=True, noise_std=0.2)

        self.pred = tf.argmax(self.logits_clear, axis=1)

        print('Done!')


    # --------------------------------------------------------------------------
    def input_graph(self):
        print('\tinput_graph')
        inputs = tf.placeholder(tf.float32, shape=[None, self.input_dim],
            name='inputs')

        labels = tf.placeholder(tf.float32, shape=[None, self.n_classes],
            name='labels')

        weight_decay = tf.placeholder(tf.float32, name='weight_decay')

        learn_rate = tf.placeholder(tf.float32, name='learn_rate')

        keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        is_training = tf.placeholder(tf.bool, name='is_training')

        return inputs, labels, weight_decay, learn_rate, keep_prob, is_training


    # --------------------------------------------------------------------------
    def encoder(self, inputs, structure, reuse, noise_std):
        # if noise_std is None no noise was used
        print('\tencoder')
        for i, layer in enumerate(structure):
            if noise_std is None: # if clear mode
                mean, var = tf.nn.moments(inputs, axes=[0])
                self.mean.append(mean)
                self.variance.append(var)
            inputs = tf.layers.dense(inputs=inputs, units=layer, activation=None,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                reuse=reuse, name='encoder'+str(i))
            if noise_std is None: # if clear mode
                self.z_clear.append(inputs)
            if noise_std is not None: # if corrapted mode
                inputs = noised_batch_norm(inputs=inputs, scale=True,
                    updates_collections=None, is_training=self.is_training,
                    reuse=reuse, scope='encoder_bn'+str(i), noise_std=noise_std)
            else: # if clear mode
                inputs = batch_norm(inputs=inputs, scale=True,
                    updates_collections=None, is_training=self.is_training,
                    reuse=reuse, scope='encoder_bn'+str(i))        
            if i < len(structure)-1:
                inputs = tf.nn.elu(inputs)
        return inputs


    # --------------------------------------------------------------------------
    def decoder(self, inputs, structure):
        print('\tdecoder')
        for i, layer in enumerate(structure):
            if i == 0:
                u = local_batch_norm(inputs)
            else:
                u = tf.layers.dense(inputs=inputs, units=layer, activation=None,
                kernel_initializer=tf.contrib.layers.xavier_initializer())
                u = local_batch_norm(u)
                z_cl = g_gauss()



    # --------------------------------------------------------------------------
    def local_batch_norm(self, inputs):
        # simple batch norm by last dimention
        sh = tf.get_shapes().as_list()
        inputs = tf.reshape(inputs, shape[-1, sh[-1]])
        mean, var = tf.nn.moments(inputs, axes=[0])
        inv = tf.rsqrt(var + 1e-6)
        return (inputs - mean)*inv


    # --------------------------------------------------------------------------
    def g_gauss(z_noised, u, size):
        "gaussian denoising function proposed in the original paper"
        wi = lambda inits, name: tf.Variable(inits * tf.ones([size]), name=name)
        a1 = wi(0., 'a1')
        a2 = wi(1., 'a2')
        a3 = wi(0., 'a3')
        a4 = wi(0., 'a4')
        a5 = wi(0., 'a5')
        a6 = wi(0., 'a6')
        a7 = wi(1., 'a7')
        a8 = wi(0., 'a8')
        a9 = wi(0., 'a9')
        a10 = wi(0., 'a10')
        mu = a1 * tf.sigmoid(a2 * u + a3) + a4 * u + a5
        v = a6 * tf.sigmoid(a7 * u + a8) + a9 * u + a10
        z_est = (z_noised - mu) * v + mu
        return z_est


    # --------------------------------------------------------------------------
    def create_cost_graph(self, labels, logits):
        print('create_cost_graph')
        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=labels,
            logits=logits))
        self.L2_loss = self.weight_decay*sum([tf.reduce_mean(tf.square(var))
            for var in tf.trainable_variables()])
        return self.cross_entropy + self.L2_loss


    # --------------------------------------------------------------------------
    def create_summary(self, labels, logits):
        with tf.name_scope('classifier'):
            self.summary.append(tf.summary.scalar('cross_entropy', self.cross_entropy))
            self.summary.append(tf.summary.scalar('L2 loss', self.L2_loss))
            precision, recall, f1, accuracy = self.get_metrics(labels, logits)
            for i in range(self.n_classes):
                self.summary.append(tf.summary.scalar('Class {} f1 score'.format(i), f1[i]))
            self.summary.append(tf.summary.scalar('Accuracy', accuracy))


    # --------------------------------------------------------------------------
    def create_optimizer_graph(self, cost):
        print('create_optimizer_graph')
        with tf.variable_scope('optimizer_graph'):
            optimizer = tf.train.AdamOptimizer(self.learn_rate)
            train = optimizer.minimize(cost)
        return train


    # --------------------------------------------------------------------------
    def train_model(self, data_loader, batch_size, weight_decay,  learn_rate_start,
        learn_rate_end, keep_prob, n_iter, save_model_every_n_iter, path_to_model):
        print('\n\t----==== Training ====----')
            
        start_time = time.time()
        for current_iter in tqdm(range(n_iter)):
            learn_rate = self.scaled_exp_decay(learn_rate_start, learn_rate_end,
                n_iter, current_iter)


            train_batch = data_loader.validation.next_batch(batch_size)
            test_batch = data_loader.test.next_batch(batch_size)
            self.train_step(train_batch[0], train_batch[1], weight_decay, learn_rate,
                keep_prob)
            if current_iter%50 == 0:
                self.save_summaries(train_batch[0], train_batch[1], weight_decay,
                    keep_prob, True, self.train_writer, current_iter)
                self.save_summaries(test_batch[0], test_batch[1],
                    weight_decay, 1, False, self.test_writer, current_iter)

            if (current_iter+1) % save_model_every_n_iter == 0:
                self.save_model(path=path_to_model, sess=self.sess, step=current_iter+1)
        self.save_model(path=path_to_model, sess=self.sess, step=current_iter+1)
        print('\nTrain finished!')
        print("Training time --- %s seconds ---" % (time.time() - start_time))

    # --------------------------------------------------------------------------
    def train_step(self, inputs, labels, weight_decay, learn_rate, keep_prob):
        feedDict = {self.inputs : inputs,
            self.labels : labels,
            self.weight_decay : weight_decay,
            self.learn_rate : learn_rate,
            self.keep_prob : keep_prob,
            self.is_training : True}
        self.sess.run(self.train, feed_dict=feedDict)


    # --------------------------------------------------------------------------
    def save_summaries(self, inputs, labels, weight_decay, keep_prob, is_training,
        writer, it):
        feedDict = {self.inputs : inputs,
            self.labels : labels,
            self.weight_decay : weight_decay,
            self.keep_prob : keep_prob,
            self.is_training : is_training}
        summary = self.sess.run(self.merged, feed_dict=feedDict)
        writer.add_summary(summary, it)



def test_classifier():
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True,
        validation_size=100)
    cl = Ladder(input_dim=784, n_classes=10, do_train=False, scope='ladder')
    # cl.train_model(data_loader=mnist, batch_size=100, weight_decay=2e-2,
    #     learn_rate_start=1e-3, learn_rate_end=1e-4, keep_prob=0.5, n_iter=200000,
    #     save_model_every_n_iter=350000, path_to_model='models/cl')

################################################################################
# TESTING
if __name__ == '__main__':
    test_classifier()




