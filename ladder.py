import time
import math

import tensorflow as tf
import numpy as np
from tqdm import tqdm
from tensorflow.contrib.layers.python.layers.layers import batch_norm

from model_abstract import Model


class Ladder(Model):

    # --------------------------------------------------------------------------
    def __init__(self, input_shape, n_classes, encoder_structure,
        decoder_structure, layer_importants, noise_std, do_train, scope):
        """ 
        Args:
            input_shape: list of integer, shape of input images
        """
        self.debug = False

        self.input_shape = input_shape
        self.n_classes = n_classes
        self.encoder_structure = encoder_structure
        self.decoder_structure = decoder_structure
        self.lambda_ = layer_importants #importanse for each layer respectively
        self.noise_std = noise_std
        self.do_train = do_train
        self.scope = scope


        self.number_of_layers = self.get_number_of_layers(encoder_structure)
        if len(self.lambda_) != (self.number_of_layers + 1):
            raise ValueError('Length of lambda_ must fit encoder architecture.\
                Expected {0}, provided {1}'.format(self.number_of_layers + 1,
                    len(self.lambda_)))

        self.z_denoised = []
        self.layer_sizes = []
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(1)
            with tf.variable_scope(scope):
                self.create_graph()
            if do_train:

                self.cost_sup = self.create_sup_cost(self.labels, self.logits_lab_noised)
                self.cost_unsup = self.create_unsup_cost(self.z_clear, self.z_denoised)
                self.train = self.create_optimizer_graph(self.cost_sup, self.cost_unsup)
                summary = self.create_summary(self.labels, self.logits_unlab_clear)
                self.train_writer, self.test_writer = self.create_summary_writers()
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

        self.images,\
        self.inputs,\
        self.labels,\
        self.weight_decay,\
        self.learn_rate,\
        self.keep_prob,\
        self.is_training = self.input_graph() # inputs shape is # b*n_f x h1 x c1

        # --- === supervised mode === ---
        #noised pass
        self.logits_lab_noised, _, _, _ = self.encoder(self.inputs,
            reuse=False, noise_std=self.noise_std, save_statistic=False)

        # --- === unsupervised mode === ---
        # clean pass
        self.logits_unlab_clear, self.mean, self.std, self.z_clear = self.encoder(
            self.images, reuse=True, noise_std=0,
            save_statistic=True)
        # noised pass
        self.logits_unlab_noised, _, _, self.z_noised = self.encoder(self.images,
             reuse=True, noise_std=self.noise_std,
            save_statistic=False)

        y = tf.nn.softmax(self.logits_unlab_noised) # b x 10              

        self.decoder(inputs=y)
    
        if self.debug:
            [print(v) for v in tf.trainable_variables()]
            print()
            print('len z_clear', len(self.z_clear))
            print('len z_noised', len(self.z_noised))
            print('len z_denoised', len(self.z_denoised))
            print('len layer_sizes', len(self.layer_sizes))
            for i in range(self.number_of_layers+1):
                print('\t ', i)
                print('mean', self.mean[i])
                print('std', self.std[i])
                print('z_noised', self.z_noised[i])
                print('z_clear', self.z_clear[i])
                print('z_denoised', self.z_denoised[i])
                print('layer_sizes', self.layer_sizes[i])
            print(self.layer_sizes)

        print('Done!')


    # --------------------------------------------------------------------------
    def input_graph(self):
        print('\tinput_graph')
        images = tf.placeholder(tf.float32, shape=[None]+self.input_shape,
            name='images')

        inputs = tf.placeholder(tf.float32, shape=[None]+self.input_shape,
            name='inputs')

        labels = tf.placeholder(tf.float32, shape=[None, self.n_classes],
            name='labels')

        weight_decay = tf.placeholder(tf.float32, name='weight_decay')

        learn_rate = tf.placeholder(tf.float32, name='learn_rate')

        keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        is_training = tf.placeholder(tf.bool, name='is_training')

        return images, inputs, labels, weight_decay, learn_rate, keep_prob, is_training


    # --------------------------------------------------------------------------
    def add_noise(self, inputs, std):
        return inputs + tf.random_normal(tf.shape(inputs),
                    stddev=std)

    # --------------------------------------------------------------------------
    def encoder(self, inputs, reuse, noise_std, save_statistic):
        print('\tencoder')
        inputs = self.add_noise(inputs, noise_std)
        mean_list, std_list, z_list = [], [], []
        L = self.number_of_layers
        
        h = inputs
        i = 0
        for layer in self.encoder_structure:
            if type(layer).__name__ == 'Reshape':
                h = tf.reshape(h, layer.shape)
                continue
            if i == 0:
                z_list.append(tf.reshape(h, [-1, h.get_shape().as_list()[-1]]))
                mean_list.append(0)
                std_list.append(1)
                if self.debug: print('input shape', h.get_shape())
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
            layer_size = z_pre.get_shape().as_list()[-1]

            mean, std = self.get_mean_std(z_pre)
            mean_list.append(mean)
            std_list.append(std)

            if save_statistic:
                z = batch_norm(inputs=z_pre, scale=False, center=False,
                    updates_collections=None, is_training=self.is_training,
                    scope='encoder_bn'+str(i), decay=0.99, trainable=False)
            else:
                z = self.local_batch_norm(z_pre)
            z = self.add_noise(z, noise_std)
            z_list.append(tf.reshape(z, [-1, layer_size]))

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
        return h, mean_list, std_list, z_list


    # --------------------------------------------------------------------------
    def decoder(self, inputs):
        print('\tdecoder')
        L = self.number_of_layers

        i = 0
        z_est = inputs
        for layer in self.decoder_structure:
            if type(layer).__name__ == 'Reshape':
                z_est = tf.reshape(z_est, layer.shape)
                continue
            if i == 0:
                u = self.local_batch_norm(z_est)
                if self.debug: print('input shape', u.get_shape())
                with tf.variable_scope("denoise_func"):
                    z_est = self.g_gauss(self.z_noised[L], u, self.n_classes)
                self.z_denoised.append(tf.reshape(
                    (z_est - self.mean[L])/(self.std[L] + 1e-5), [-1, self.n_classes]))
                self.layer_sizes.append(z_est.get_shape().as_list()[-1])
            if type(layer).__name__ == 'ConvoLayer':
                u = tf.layers.conv2d(z_est, layer.filters, layer.kernel_size,
                    layer.strides, layer.padding,
                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                    name='decoder_layer'+str(i))
            elif type(layer).__name__ == 'DeConvoLayer':
                u = tf.layers.conv2d_transpose(z_est, layer.filters,
                    layer.kernel_size, layer.strides, layer.padding,
                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                    name='decoder_layer'+str(i))
            elif type(layer).__name__ == 'DenseLayer':
                u = tf.layers.dense(inputs=z_est, units=layer.units, activation=None,
                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                    name='decoder_layer'+str(i))
            else:
                raise NotImplementedError('Type {} of layer is not implemented'.format(
                    type(layer).__name__))
            layer_size = u.get_shape().as_list()[-1]

            u = self.local_batch_norm(u)
            with tf.variable_scope("denoise_func_layer{}".format(i)):
                z_est = self.g_gauss(self.z_noised[L-i-1], u, layer_size)
            z_denoised = tf.reshape((z_est - self.mean[L-i-1])/(self.std[L-i-1] + 1e-5),
                [-1, layer_size])
            self.z_denoised.append(z_denoised)
            self.layer_sizes.append(layer_size)
            if self.debug: print('layer {} shape'.format(i), u.get_shape())
            i += 1 # increase number of layer
        self.layer_sizes = list(reversed(self.layer_sizes))    
        self.z_denoised = list(reversed(self.z_denoised))


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
    def g_gauss(self, z_noised, u, size):
        z_noised = tf.reshape(z_noised, tf.shape(u))
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
    def create_unsup_cost(self, z_clear, z_denoised):
        print('create_unsup_cost')
        self.denoise_loss_list = []
        for lamb, layer_size, z_cl, z_denois in zip(self.lambda_,
            self.layer_sizes, z_clear, z_denoised):
            self.denoise_loss_list.append(tf.reduce_mean(tf.reduce_sum(
                tf.square(z_cl-z_denois), 1))/layer_size*lamb)
        self.denoise_loss = tf.add_n(self.denoise_loss_list)
        return self.denoise_loss


    # --------------------------------------------------------------------------
    def create_summary(self, labels, logits):
        summary = []
        summary.append(tf.summary.scalar('cross_entropy', self.cross_entropy))
        summary.append(tf.summary.scalar('L2 loss', self.L2_loss))
        summary.append(tf.summary.scalar('denoise_loss', self.denoise_loss))

        precision, recall, self.f1, self.accuracy = self.get_metrics(labels, logits)
        for i in range(self.n_classes):
            summary.append(tf.summary.scalar('Class {} f1 score'.format(i), self.f1[i]))
        summary.append(tf.summary.scalar('Accuracy', self.accuracy))

        if self.debug:
            with tf.name_scope('Variables_grads'):
                for var in tf.trainable_variables():
                    summary.append(tf.summary.scalar(var.name, tf.reduce_mean(var)))

            with tf.name_scope("reconstract_loss_per_layer"):
                for i, loss in enumerate(self.denoise_loss_list):
                    summary.append(tf.summary.scalar('layer_'.format(i), loss))

            grad_sup = self.optimizer.compute_gradients(self.cost_sup)
            grad_unsup = self.optimizer.compute_gradients(self.cost_unsup)

            grad_sup_mean = tf.reduce_mean([tf.reduce_mean(tf.abs(t)) for t,n in grad_sup\
                    if t is not None])
            grad_unsup_mean = tf.reduce_mean([tf.reduce_mean(tf.abs(t)) for t,n in grad_unsup\
                    if t is not None])
            summary.append(tf.summary.scalar('grad_sup', grad_sup_mean))
            summary.append(tf.summary.scalar('grad_unsup', grad_unsup_mean))

            with tf.name_scope('supervised_gradients'):
                for grad, n in grad_sup:
                    summary.append(tf.summary.scalar(n.name, tf.reduce_mean(grad)))

            with tf.name_scope('unsupervised_gradients'):
                for grad, n in grad_unsup:
                    summary.append(tf.summary.scalar(n.name, tf.reduce_mean(grad)))
        return summary


    # --------------------------------------------------------------------------
    def create_optimizer_graph(self, cost_sup, cost_unsup):
        print('create_optimizer_graph')
        with tf.variable_scope('optimizer_graph'):
            self.optimizer = tf.train.AdamOptimizer(self.learn_rate)
            grad_var = self.optimizer.compute_gradients(cost_sup + cost_unsup)
            # grad_var = [(tf.clip_by_value(g, -1, 1), v) for g,v in grad_var]
            train = self.optimizer.apply_gradients(grad_var)
        return train


    # --------------------------------------------------------------------------
    def train_step(self, inputs_lab, inputs_unlab, labels, weight_decay,
        learn_rate, keep_prob):
        # inputs_lab = np.reshape(inputs_lab, [-1]+self.input_shape)
        # inputs_unlab = np.reshape(inputs_unlab, [-1]+self.input_shape)

        feedDict = {self.images : inputs_unlab,
            self.inputs : inputs_lab,
            self.labels : labels,
            self.weight_decay : weight_decay,
            self.learn_rate : learn_rate,
            self.keep_prob : keep_prob,
            self.is_training : True}
        self.sess.run(self.train, feed_dict=feedDict)


    # --------------------------------------------------------------------------
    def save_summaries(self, inputs_lab, inputs_unlab, labels, weight_decay,
        keep_prob, is_training, writer, it):
        # inputs_lab = np.reshape(inputs_lab, [-1]+self.input_shape)
        # inputs_unlab = np.reshape(inputs_unlab, [-1]+self.input_shape)

        feedDict = {self.images : inputs_unlab,
            self.inputs : inputs_lab,
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
    def train_model(self, image_provider, labeled_data_loader, test_data_loader,
        batch_size, weight_decay,  learn_rate_start,
        learn_rate_end, keep_prob, n_iter, save_model_every_n_iter, path_to_model):

        print('\n\t----==== Training ====----')
        start_time = time.time()
        for current_iter in tqdm(range(n_iter)):
            learn_rate = self.scaled_exp_decay(learn_rate_start, learn_rate_end,
                n_iter, current_iter)

            images = image_provider(batch_size)
            train_batch = labeled_data_loader.next_batch(batch_size)
            test_batch = test_data_loader.next_batch(batch_size)
            self.train_step(train_batch[0], images, train_batch[1], weight_decay,
                learn_rate, keep_prob)
            if current_iter%10 == 0:
                self.save_summaries(train_batch[0], train_batch[0], train_batch[1],
                    weight_decay, keep_prob, True, self.train_writer, current_iter)
                self.save_summaries(test_batch[0], test_batch[0], test_batch[1],
                    weight_decay, 1, False, self.test_writer, current_iter)

            if (current_iter+1) % save_model_every_n_iter == 0:
                self.save_model(path=path_to_model, sess=self.sess, step=current_iter+1)

            if current_iter%5000 == 0:
                self.evaluate_metrics(test_data_loader.images,
                    test_data_loader.labels, current_iter)
        self.save_model(path=path_to_model, sess=self.sess, step=current_iter+1)
        print('\nTrain finished!')
        print('Final metrix')
        self.evaluate_metrics(test_data_loader.images,
                    test_data_loader.labels, current_iter)
        print("Training time --- %s seconds ---" % (time.time() - start_time))




