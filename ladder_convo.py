import time
import math

import tensorflow as tf
import numpy as np
from tqdm import tqdm
from tensorflow.contrib.layers.python.layers.layers import batch_norm

from model_abstract import Model
from param import encoder as encoder_structure
from param import decoder as decoder_structure
from param import layer_importants


class ConvoLadder(Model):

    # --------------------------------------------------------------------------
    def __init__(self, input_shape, n_classes, do_train, scope):
        """ 
        Args:
            input_shape: list of integer, shape of input images
        """

        self.input_shape = input_shape
        self.n_classes = n_classes
        self.do_train = do_train


        self.noise_std = 0.3
        self.number_of_layers = self.get_number_of_layers(encoder_structure)
        self.lambda_ = layer_importants #importanse for each layer respectively
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
                self.train_writer, self.test_writer = self.create_summary_writers(
                    'summary/ladder_convo')
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
        # clean pass
        # _, _, _, _ = self.encoder(self.inputs,
        #     reuse=False, noise_std=0, save_statistic=True)
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


        y = encoder_structure[-1].activation(self.logits_unlab_noised) # b x 10
        # y = tf.Print(y, [y], message='y', first_n=10)

        self.decoder(inputs=y)
    
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
        
        z_list.append(tf.reshape(inputs, [-1, self.input_shape[-1]]))
        mean_list.append(0)
        std_list.append(1)

        h = inputs
        i = 0
        for layer in encoder_structure:
            print('\n', i, layer)
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
            print('z_pre', z_pre)
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
                print('activate')
                h = layer.activation(h)
            i += 1 # increase number of layer

        print('encoder final shape', h)
        return h, mean_list, std_list, z_list


    # --------------------------------------------------------------------------
    def decoder(self, inputs):
        print('\tdecoder')
        L = self.number_of_layers

        u = self.local_batch_norm(inputs)
        z_est = self.g_gauss(self.z_noised[L], u, self.n_classes)
        self.z_denoised.append(tf.reshape(
            (z_est - self.mean[L])/(self.std[L] + 1e-5), [-1, self.n_classes]))
        self.layer_sizes.append(z_est.get_shape().as_list()[-1])
        
        i = 0
        for layer in decoder_structure:
            print('\n', i, layer)
            if type(layer).__name__ == 'Reshape':
                z_est = tf.reshape(z_est, layer.shape)
                continue
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
            print('u',u)
            layer_size = u.get_shape().as_list()[-1]

            u = self.local_batch_norm(u)
            print('z_noised', self.z_noised[L-i-1])
            with tf.variable_scope("denoise_func_layer{}".format(i)):
                z_est = self.g_gauss(self.z_noised[L-i-1], u, layer_size)
            print('z_est',z_est)
            z_denoised = tf.reshape((z_est - self.mean[L-i-1])/(self.std[L-i-1] + 1e-5),
                [-1, layer_size])
            print('z_denoised', z_denoised)
            print('self.mean[L-i-1]', self.mean[L-i-1])
            print('self.std[L-i-1]', self.std[L-i-1])
            self.z_denoised.append(z_denoised)
            self.layer_sizes.append(layer_size)
            i += 1
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
        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=labels,
            logits=logits))
        self.L2_loss = self.weight_decay*sum([tf.reduce_mean(tf.square(var))
            for var in tf.trainable_variables()])
        # self.cross_entropy = tf.Print(self.cross_entropy, [self.cross_entropy],
        #     message='self.cross_entropy', first_n=10)
        # self.L2_loss = tf.Print(self.L2_loss, [self.L2_loss],
        #     message='self.L2_loss', first_n=10)

        return self.cross_entropy + self.L2_loss


    # --------------------------------------------------------------------------
    def create_unsup_cost(self, z_clear, z_denoised):
        print('create_unsup_cost')      
        self.denoise_loss_list = []
        for lamb, layer_width, z_cl, z_denois in zip(self.lambda_,
            self.layer_sizes, z_clear, z_denoised):
            # print('\n', lamb, layer_width, z_cl, z_denois)

            self.denoise_loss_list.append(1e-10*lamb/layer_width*tf.reduce_mean(tf.square(
                tf.norm(z_cl-z_denois, axis=1))))

        self.denoise_loss = tf.add_n(self.denoise_loss_list)
        # self.denoise_loss = tf.Print(self.denoise_loss, [self.denoise_loss],
        #     message='denoise_loss', first_n=10)
        return self.denoise_loss


    # --------------------------------------------------------------------------
    def create_summary(self, labels, logits):
        summary = []
        summary.append(tf.summary.scalar('cross_entropy', self.cross_entropy))
        summary.append(tf.summary.scalar('L2 loss', self.L2_loss))
        summary.append(tf.summary.scalar('denoise_loss', self.denoise_loss))

        precision, recall, f1, self.accuracy = self.get_metrics(labels, logits)
        for i in range(self.n_classes):
            summary.append(tf.summary.scalar('Class {} f1 score'.format(i), f1[i]))
        summary.append(tf.summary.scalar('Accuracy', self.accuracy))

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
            grad_var = [(tf.clip_by_value(g, -1, 1), v) for g,v in grad_var]
            train = self.optimizer.apply_gradients(grad_var)




        return train


    # --------------------------------------------------------------------------
    def train_step(self, inputs_lab, inputs_unlab, labels, weight_decay,
        learn_rate, keep_prob):
        inputs_lab = np.reshape(inputs_lab, [-1]+self.input_shape)
        inputs_unlab = np.reshape(inputs_unlab, [-1]+self.input_shape)

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
        inputs_lab = np.reshape(inputs_lab, [-1]+self.input_shape)
        inputs_unlab = np.reshape(inputs_unlab, [-1]+self.input_shape)

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
    def train_model(self, image_provider, labeled_data_loader, test_data_loader,
        batch_size, weight_decay,  learn_rate_start,
        learn_rate_end, keep_prob, n_iter, save_model_every_n_iter, path_to_model):

        def get_acc():
            inp = np.reshape(test_data_loader.images, [-1] + self.input_shape)
            # inp = test_data_loader.images
            acc = self.sess.run(self.accuracy, {self.images : inp,
            self.labels : test_data_loader.labels, self.is_training : False})
            return acc

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
            if current_iter%10 == 0:
                self.save_summaries(train_batch[0], train_batch[0], train_batch[1],
                    weight_decay, keep_prob, True, self.train_writer, current_iter)
                self.save_summaries(test_batch[0], test_batch[0], test_batch[1],
                    weight_decay, 1, False, self.test_writer, current_iter)

            if (current_iter+1) % save_model_every_n_iter == 0:
                self.save_model(path=path_to_model, sess=self.sess, step=current_iter+1)

            # if current_iter%5000 == 0:
            #     print('Iteration {0}, accuracy {1}'.format(current_iter+1, get_acc()))
        self.save_model(path=path_to_model, sess=self.sess, step=current_iter+1)
        print('\nTrain finished!')
        print('Final accuracy {0}'.format(get_acc()))
        print("Training time --- %s seconds ---" % (time.time() - start_time))


def list_mult(l):
    product = 1
    for x in l:
        product *= x
    return product

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
    weight_decay = 2e-5
    n_iter = 200000
    learn_rate_start = 1e-2
    learn_rate_end = 1e-4
    keep_prob = 0.5
    save_model_every_n_iter = 15000
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

    cl = ConvoLadder(input_shape=[28,28,1], n_classes=10, do_train=True, scope='ladder')
    cl.train_model(image_provider, labeled_data_loader, test_data_loader,
        batch_size, weight_decay, learn_rate_start, learn_rate_end, keep_prob,
        n_iter, save_model_every_n_iter, path_to_model)

################################################################################
# TESTING
if __name__ == '__main__':
    test_classifier()




