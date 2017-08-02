import time
import math

import tensorflow as tf
import numpy as np
from tqdm import tqdm

from my_layers import batch_norm as my_batch_norm
from model_abstract import Model


class Ladder(Model):

    # --------------------------------------------------------------------------
    def __init__(self, input_dim, n_classes, do_train, scope):

        self.input_dim = input_dim
        self.n_classes = n_classes
        self.do_train = do_train

        self.structure = [1000, 500, 250, 250, 250, self.n_classes]
        self.lambda_ = [1000, 10, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1] #importanse for each layer respectively
        self.noise_std = 0.2

        self.mean, self.std, self.z_noised, self.z_clear, self.z_denoised = [], [], [], [], []
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(1)
            with tf.variable_scope(scope):
                self.create_graph()
            if do_train:
                self.cost_sup, self.cost_unsup = self.create_cost_graph(self.labels,
                    self.logits_noised, self.z_clear, self.z_denoised)
                s_sup, s_unsup = self.create_summary(self.labels, self.logits_clear)
                self.train_sup, self.train_unsup = self.create_optimizer_graph(
                    self.cost_sup, self.cost_unsup)
                self.train_writer, self.test_writer = self.create_summary_writers(
                    'summary/ladder')
                self.merged_sup = tf.summary.merge(s_sup)
                self.merged_unsup = tf.summary.merge(s_unsup)

            self.sess = self.create_session()
            self.sess.run(tf.global_variables_initializer())
            self.stored_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                scope=scope)
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
            structure=self.structure, reuse=False, noise_std=None)
        noised_inputs = self.inputs+tf.random_normal(tf.shape(self.inputs),
            stddev=self.noise_std)
        self.logits_noised = self.encoder(noised_inputs,
            structure=self.structure, reuse=True, noise_std=self.noise_std)
        self.pred = tf.argmax(self.logits_clear, axis=1)

        # [print(v) for v in tf.trainable_variables()]

        y = tf.nn.softmax(self.logits_noised)

        self.decoder(inputs=y, structure=self.structure)
        
        # print()
        # print('len z_clear', len(self.z_clear))
        # print('len z_noised', len(self.z_noised))
        # print('len z_denoised', len(self.z_denoised))
        # for i in range(len(self.structure)+1):
        #     print('\t ', i)
        #     print('mean', self.mean[i])
        #     print('std', self.std[i])
        #     print('z_noised', self.z_noised[i])
        #     print('z_clear', self.z_clear[i])
        #     print('z_denoised', self.z_denoised[i])

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
        if noise_std is None: # if clear mode
            self.z_clear.append(inputs)
        else: # noised mode
            self.z_noised.append(inputs)
        L = len(structure)
        for i, layer in enumerate(structure):
            if noise_std is None: # if clear mode
                mean, var = tf.nn.moments(inputs, axes=[0])
                self.mean.append(mean)
                self.std.append(tf.sqrt(var))
            inputs = tf.layers.dense(inputs=inputs, units=layer, activation=None,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                reuse=reuse, name='encoder'+str(i))
            if i == (L-1):
                mean, var = tf.nn.moments(inputs, axes=[0])
                self.mean.append(mean)
                self.std.append(tf.sqrt(var))
            if noise_std is None: # if clear mode
                z, inputs = my_batch_norm(inputs=inputs, scale=True,
                    updates_collections=None, is_training=self.is_training,
                    reuse=reuse, scope='encoder_bn'+str(i), noise_std=0)
                self.z_clear.append(z)
            else: # noised mode
                z, inputs = my_batch_norm(inputs=inputs, scale=True,
                    updates_collections=None, is_training=self.is_training,
                    reuse=reuse, scope='encoder_bn'+str(i), noise_std=noise_std)        
                self.z_noised.append(z)
            if i < len(structure)-1:
                inputs = tf.nn.relu(inputs)
        return inputs


    # --------------------------------------------------------------------------
    def decoder(self, inputs, structure):
        print('\tdecoder')
        structure = list(reversed(structure[:-1]))
        structure.append(self.input_dim)
        # print('structure', structure)
        L = len(structure)

        u = self.local_batch_norm(inputs)
        z_denoised = self.g_gauss(self.z_noised[L], u, self.n_classes)
        z_denoised = (z_denoised - self.mean[L])/(self.std[L] + 1e-3)
        self.z_denoised.append(z_denoised)

        for i, layer in enumerate(structure):
            # print(i, layer)
            u = tf.layers.dense(inputs=self.z_clear[L-i-1], units=layer, activation=None,
                kernel_initializer=tf.contrib.layers.xavier_initializer())
            u = self.local_batch_norm(u)
            # print('u',u)
            # print('z_noised', self.z_noised[L-i-1])
            z_denoised = self.g_gauss(self.z_noised[L-i-1], u, layer)
            # print('z_denoised',z_denoised)
            z_denoised = (z_denoised - self.mean[L-i-1])/(self.std[L-i-1] + 1e-3)
            self.z_denoised.append(z_denoised)
        self.z_denoised = list(reversed(self.z_denoised))



    # --------------------------------------------------------------------------
    def local_batch_norm(self, inputs):
        # simple batch norm by last dimention
        sh = inputs.get_shape().as_list()
        inputs = tf.reshape(inputs, shape=[-1, sh[-1]])
        mean, var = tf.nn.moments(inputs, axes=[0])
        inv = tf.rsqrt(var + 1e-3)
        return (inputs - mean)*inv


    # --------------------------------------------------------------------------
    def g_gauss(self, z_noised, u, size):
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
    def create_cost_graph(self, labels, logits, z_clear, z_denoised):
        print('create_cost_graph')

        # supervised loss
        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=labels,
            logits=logits))
        self.L2_loss = self.weight_decay*sum([tf.reduce_mean(tf.square(var))
            for var in tf.trainable_variables()])

        # unsupervised loss
        denoise_loss = []
        for lamb, layer_width, z_cl, z_denois in zip(self.lambda_,
            [self.input_dim]+self.structure, z_clear, z_denoised):
            # print('\n', lamb, layer_width, z_cl, z_denois)

            denoise_loss.append(lamb/layer_width*tf.reduce_mean(tf.square(
                tf.norm(z_cl-z_denois, axis=1))))

        self.denoise_loss = tf.add_n(denoise_loss)

        cost_sup = self.cross_entropy + self.L2_loss
        cost_unsup = self.denoise_loss + self.L2_loss

        return cost_sup, cost_unsup


    # --------------------------------------------------------------------------
    def create_summary(self, labels, logits):
        s_sup, s_unsup = [], []
        s_sup.append(tf.summary.scalar('cross_entropy', self.cross_entropy))
        s_sup.append(tf.summary.scalar('L2 loss', self.L2_loss))

        precision, recall, f1, accuracy = self.get_metrics(labels, logits)
        for i in range(self.n_classes):
            s_sup.append(tf.summary.scalar('Class {} f1 score'.format(i), f1[i]))
        s_sup.append(tf.summary.scalar('Accuracy', accuracy))
        
        s_unsup.append(tf.summary.scalar('denoise_loss', self.denoise_loss))
        return s_sup, s_unsup


    # --------------------------------------------------------------------------
    def create_optimizer_graph(self, cost_sup, cost_unsup):
        print('create_optimizer_graph')
        with tf.variable_scope('optimizer_graph'):
            optimizer = tf.train.AdamOptimizer(self.learn_rate)
            train_sup = optimizer.minimize(cost_sup)
            train_unsup = optimizer.minimize(cost_unsup)
        return train_sup, train_unsup


    # --------------------------------------------------------------------------
    def train_model(self, image_provider, labeled_data_loader, test_data_loader,
        batch_size, weight_decay,  learn_rate_start,
        learn_rate_end, keep_prob, n_iter, save_model_every_n_iter, path_to_model):
        print('\n\t----==== Training ====----')
            
        start_time = time.time()
        for current_iter in tqdm(range(n_iter)):
            learn_rate = self.scaled_exp_decay(learn_rate_start, learn_rate_end,
                n_iter, current_iter)

            images = image_provider.next_batch(batch_size)
            train_batch = labeled_data_loader.next_batch(batch_size)
            test_batch = test_data_loader.next_batch(batch_size)
            self.train_step(train_batch[0], images, train_batch[1], weight_decay,
                learn_rate, keep_prob)
            if current_iter%50 == 0:
                self.save_summaries(train_batch[0], images, train_batch[1],
                    weight_decay, keep_prob, True, self.train_writer, current_iter)
                self.save_summaries(test_batch[0], test_batch[0], test_batch[1],
                    weight_decay, 1, False, self.test_writer, current_iter)

            if (current_iter+1) % save_model_every_n_iter == 0:
                self.save_model(path=path_to_model, sess=self.sess, step=current_iter+1)
        self.save_model(path=path_to_model, sess=self.sess, step=current_iter+1)
        print('\nTrain finished!')
        print("Training time --- %s seconds ---" % (time.time() - start_time))

    # --------------------------------------------------------------------------
    def train_step(self, inputs_lab, inputs_unlab, labels, weight_decay,
        learn_rate, keep_prob):

        feedDict = {self.inputs : inputs_lab,
            self.labels : labels,
            self.weight_decay : weight_decay,
            self.learn_rate : learn_rate,
            self.keep_prob : keep_prob,
            self.is_training : True}
        self.sess.run(self.train_sup, feed_dict=feedDict)

        feedDict[self.inputs] = inputs_unlab
        self.sess.run(self.train_unsup, feed_dict=feedDict)


    # --------------------------------------------------------------------------
    def save_summaries(self, inputs_lab, inputs_unlab, labels, weight_decay,
        keep_prob, is_training, writer, it):

        feedDict = {self.inputs : inputs_lab,
            self.labels : labels,
            self.weight_decay : weight_decay,
            self.keep_prob : keep_prob,
            self.is_training : is_training}
        summary = self.sess.run(self.merged_sup, feed_dict=feedDict)
        writer.add_summary(summary, it)

        feedDict[self.inputs] = inputs_unlab
        summary = self.sess.run(self.merged_unsup, feed_dict=feedDict)
        writer.add_summary(summary, it)

def test_classifier():
    from tensorflow.examples.tutorials.mnist import input_data

    class ImageProvider:
        def __init__(self):
            self.mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
            print('total number of images', self.mnist.train.num_examples)
        def next_batch(self, batch_size):
            return self.mnist.train.next_batch(batch_size)[0]

    labeled_size = 100
    batch_size = 100
    weight_decay = 2e-2
    n_iter = 200000
    learn_rate_start = 1e-3
    learn_rate_end = 1e-4
    keep_prob = 0.5
    save_model_every_n_iter = 35000
    path_to_model = 'models/ladder'

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True,
        validation_size=labeled_size)
    labeled_data_loader = mnist.validation
    print('total number of labeled data', labeled_data_loader.num_examples)
    l = labeled_data_loader.labels
    print('Distribution of labeled data:')
    [print('class_{0} = {1}'.format(i,v)) for i,v in enumerate(np.sum(l,0))]
    test_data_loader = mnist.test
    print('total number of test data', test_data_loader.num_examples)
    image_provider = ImageProvider()

    cl = Ladder(input_dim=784, n_classes=10, do_train=True, scope='ladder')
    cl.train_model(image_provider, labeled_data_loader, test_data_loader,
        batch_size, weight_decay, learn_rate_start, learn_rate_end, keep_prob,
        n_iter, save_model_every_n_iter, path_to_model)

################################################################################
# TESTING
if __name__ == '__main__':
    test_classifier()




