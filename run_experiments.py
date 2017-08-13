from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet
import numpy as np

import tools
import param
from ladder import Ladder
import my_mnist

class ImageProvider:
    def __init__(self, images, labels):
        self.loader = DataSet(images, labels, one_hot=True)
    def next_batch(self, batch_size):
        return self.loader.next_batch(batch_size)[0]

train_im, train_l, test_im, test_l = my_mnist.read_data_sets("MNIST_data/")

# class_distrib = {i:i+1 for i in range(10)}
class_distrib = {0:18, 1:16, 2:15, 3:13, 4:11, 5:9, 6:7, 7:5, 8:4, 9:2}
lab_im, lab_t, rest_im, rest_t = tools.split_data_set(class_distrib, train_im,
    train_l, verbose=True)

# class_distrib = {i:5400 for i in range(10)}
class_distrib = {0:5400, 1:4860, 2:4320, 3:3780, 4:3240, 5:2700, 6:2160,
                 7:1620, 8:1080, 9:540}
unlab_im, unlab_t, rest_im, rest_t = tools.split_data_set(class_distrib, rest_im,
    rest_t)
print('\n labeled summary:')
tools.print_stat(lab_t)
print('\n unlabeled summary:')
tools.print_stat(unlab_t)

labeled_data_loader = DataSet(lab_im, lab_t, one_hot=True)
test_data_loader = DataSet(test_im, test_l, one_hot=True)

image_provider = ImageProvider(np.vstack((lab_im, unlab_im)),
                              np.vstack((lab_t, unlab_t)))


batch_size = 100
weight_decay = 2e-3
n_iter = 100000
learn_rate_start = 1e-2
learn_rate_end = 1e-4
noise_std = 0.3
keep_prob = 1
save_model_every_n_iter = 15000
path_to_model = 'models/ladder'


cl = Ladder(input_shape=[784], n_classes=10, encoder_structure=param.dense_encoder,
    decoder_structure=param.dense_decoder, layer_importants=param.dense_layer_importants,
    noise_std=noise_std, do_train=True, scope='ladder')
cl.train_model(image_provider, labeled_data_loader, test_data_loader,
    batch_size, weight_decay, learn_rate_start, learn_rate_end, keep_prob,
    n_iter, save_model_every_n_iter, path_to_model)




