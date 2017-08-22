import time
import math

import tensorflow as tf
import numpy as np
from tqdm import tqdm
from tensorflow.contrib.layers.python.layers.layers import batch_norm

from model_abstract import Model


class ConvoNet(Model):

    # --------------------------------------------------------------------------
    def __init__(self, input_shape, n_classes, encoder_structure,
        do_train, scope):
        """ 
        Args:
            input_shape: list of integer, shape of input images
        """
        self.debug = False

        self.input_shape = input_shape
        self.n_classes = n_classes
        self.encoder_structure = encoder_structure
        self.do_train = do_train
        self.scope = scope


        self.number_of_layers = self.get_number_of_layers(encoder_structure)

        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(1)
            with tf.variable_scope(scope):
                self.create_graph()
            if do_train:

                self.cost_sup = self.create_sup_cost(self.labels, self.logits)
                self.train = self.create_optimizer_graph(self.cost_sup)
                summary = self.create_summary(self.labels, self.logits)
                self.train_writer, self.test_writer = self.create_summary_writers('summary/convo_net_4000_labels')
                self.train_writer.add_graph(self.graph)
                self.merged = tf.summary.merge(summary)

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

        # --- === supervised mode === ---
        #noised pass
        self.logits = self.encoder(self.inputs, reuse=False)

        y = tf.nn.softmax(self.logits) # b x 10              

        print('Done!')


    # --------------------------------------------------------------------------
    def input_graph(self):
        print('\tinput_graph')

        inputs = tf.placeholder(tf.float32, shape=[None]+self.input_shape,
            name='inputs')

        labels = tf.placeholder(tf.float32, shape=[None, self.n_classes],
            name='labels')

        weight_decay = tf.placeholder(tf.float32, name='weight_decay')

        learn_rate = tf.placeholder(tf.float32, name='learn_rate')

        keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        is_training = tf.placeholder(tf.bool, name='is_training')

        return inputs, labels, weight_decay, learn_rate, keep_prob, is_training


    # --------------------------------------------------------------------------
    def encoder(self, inputs, reuse):
        print('\tencoder')
        L = self.number_of_layers
        
        h = inputs
        i = 0
        for layer in self.encoder_structure:
            if type(layer).__name__ == 'Reshape':
                h = tf.reshape(h, layer.shape)
                continue
            if type(layer).__name__ == 'ConvoLayer':
                z_pre = tf.layers.conv2d(h, layer.filters, layer.kernel_size,
                    layer.strides, layer.padding,
                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                    reuse=reuse, name='encoder_layer'+str(i))
            elif type(layer).__name__ == 'MaxPool':
                z_pre = tf.layers.max_pooling2d(h, layer.pool_size,
                    layer.strides, layer.padding)
            elif type(layer).__name__ == 'AvrPool':
                z_pre = tf.layers.average_pooling2d(h, layer.pool_size,
                    layer.strides, layer.padding)
            elif type(layer).__name__ == 'DenseLayer':
                z_pre = tf.layers.dense(inputs=h, units=layer.units, activation=None,
                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                    reuse=reuse, name='encoder_layer'+str(i))
            else:
                raise NotImplementedError('Type {} of layer is not implemented'.format(
                    type(layer).__name__))

            z = batch_norm(inputs=z_pre, scale=False, center=False,
                updates_collections=None, is_training=self.is_training,
                scope='encoder_bn'+str(i), decay=0.99, trainable=False)

            layer_size = z_pre.get_shape().as_list()[-1]
            with tf.variable_scope('gamma_beta', reuse=reuse):
                beta = tf.get_variable(name='beta_layer'+str(i),
                    initializer=tf.zeros([layer_size]))
                gamma = tf.get_variable(name='gamma_layer'+str(i),
                    initializer=tf.ones([layer_size]))
            h = gamma*(z+beta)
            if (type(layer).__name__ == 'ConvoLayer' or
                type(layer).__name__ == 'DenseLayer') and i < (L-1):
                h = layer.activation(h)
            if self.debug: print('layer {} shape'.format(i), h.get_shape())
            i += 1 # increase number of layer
        return h


    # --------------------------------------------------------------------------
    def get_mean_std(self, inputs):
        sh = inputs.get_shape().as_list()
        inputs = tf.reshape(inputs, shape=[-1, sh[-1]])
        mean, var = tf.nn.moments(inputs, axes=[0])
        std = tf.sqrt(var + 1e-5)
        return mean, std


    # --------------------------------------------------------------------------
    def local_batch_norm(self, inputs):
        # simple batch norm by last dimention
        mean, std = self.get_mean_std(inputs)        
        return (inputs - mean)/(std + 1e-5)


    # --------------------------------------------------------------------------
    def create_sup_cost(self, labels, logits):
        print('create_sup_cost')
        # pred = tf.nn.softmax(logits)
        # self.cross_entropy = tf.reduce_mean(tf.reduce_sum(
        #     -labels*tf.log(tf.clip_by_value(pred, 1e-3, 1-1e-3))*np.array(
        #     [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), 1))
        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=labels,
            logits=logits))
        self.L2_loss = self.weight_decay*sum([tf.reduce_mean(tf.square(var))
            for var in tf.trainable_variables()])
        return self.cross_entropy + self.L2_loss


    # --------------------------------------------------------------------------
    def create_summary(self, labels, logits):
        summary = []
        summary.append(tf.summary.scalar('cross_entropy', self.cross_entropy))
        summary.append(tf.summary.scalar('L2 loss', self.L2_loss))

        precision, recall, self.f1, self.accuracy = self.get_metrics(labels, logits)
        for i in range(self.n_classes):
            summary.append(tf.summary.scalar('Class {} f1 score'.format(i), self.f1[i]))
        summary.append(tf.summary.scalar('Accuracy', self.accuracy))

        return summary


    # --------------------------------------------------------------------------
    def create_optimizer_graph(self, cost_sup):
        print('create_optimizer_graph')
        with tf.variable_scope('optimizer_graph'):
            self.optimizer = tf.train.AdamOptimizer(self.learn_rate)
            grad_var = self.optimizer.compute_gradients(cost_sup)
            # grad_var = [(tf.clip_by_value(g, -1, 1), v) for g,v in grad_var]
            train = self.optimizer.apply_gradients(grad_var)
        return train


    # --------------------------------------------------------------------------
    def train_step(self, inputs, labels, weight_decay, learn_rate, keep_prob):

        feedDict = {
            self.inputs : inputs,
            self.labels : labels,
            self.weight_decay : weight_decay,
            self.learn_rate : learn_rate,
            self.keep_prob : keep_prob,
            self.is_training : True}
        self.sess.run(self.train, feed_dict=feedDict)


    # --------------------------------------------------------------------------
    def save_summaries(self, inputs, labels, weight_decay,
        keep_prob, is_training, writer, it):

        feedDict = {
            self.inputs : inputs,
            self.labels : labels,
            self.weight_decay : weight_decay,
            self.keep_prob : keep_prob,
            self.is_training : is_training}
        summary = self.sess.run(self.merged, feed_dict=feedDict)
        writer.add_summary(summary, it)


    # --------------------------------------------------------------------------
    def get_number_of_layers(self, encoder_structure):
        number_of_layers = 0
        for layer in encoder_structure:
            if type(layer).__name__ != 'Reshape':
                number_of_layers += 1
        return number_of_layers


    # --------------------------------------------------------------------------
    def evaluate_metrics(self, images, labels, iteration):
        acc, f1 = self.sess.run([self.accuracy, self.f1], {self.images : images,
            self.labels : labels, self.is_training : False})
        print('\n---=== Iteration {} ===---'.format(iteration))
        print('accuracy = {}'.format(acc))
        [print('class {0}, f1 = {1}'.format(i,f)) for i,f in enumerate(f1)]
        with open(self.scope+'_log.csv', 'a') as f:
            f.write('Iteration {}\n'.format(iteration))
            f.write('accuracy,{}\n'.format(acc))
            f.write('class,f1-score\n')
            [f.write('{},{}\n'.format(i,fscore)) for i,fscore in enumerate(f1)]


    # --------------------------------------------------------------------------
    def train_model(self, train_data_loader, test_data_loader,
        batch_size, weight_decay,  learn_rate_start,
        learn_rate_end, keep_prob, n_iter, save_model_every_n_iter, path_to_model):

        print('\n\t----==== Training ====----')
        start_time = time.time()
        for current_iter in tqdm(range(n_iter)):
            learn_rate = self.scaled_exp_decay(learn_rate_start, learn_rate_end,
                n_iter, current_iter)

            train_batch = train_data_loader.next_batch(batch_size)
            test_batch = test_data_loader.next_batch(batch_size)
            self.train_step(train_batch[0], train_batch[1], weight_decay,
                learn_rate, keep_prob)
            if current_iter%10 == 0:
                self.save_summaries(train_batch[0], train_batch[1],
                    weight_decay, keep_prob, True, self.train_writer, current_iter)
                self.save_summaries(test_batch[0], test_batch[1],
                    weight_decay, 1, False, self.test_writer, current_iter)

            if (current_iter+1) % save_model_every_n_iter == 0:
                self.save_model(path=path_to_model, sess=self.sess, step=current_iter+1)

        self.save_model(path=path_to_model, sess=self.sess, step=current_iter+1)
        print('\nTrain finished!')
        print("Training time --- %s seconds ---" % (time.time() - start_time))




