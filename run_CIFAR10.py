import tensorflow as tf

import tools
import ladder_tools
import param
from ladder import Ladder

class_distrib_lab = {i:400 for i in range(10)}

image_provider, labeled_data_loader, test_data_loader = ladder_tools.get_cifar10_loaders(
    class_distrib_lab)


batch_size = 100
weight_decay = 2e-3
n_iter = 100000
learn_rate_start = 1e-2
learn_rate_end = 1e-4
noise_std = 0.3
keep_prob = 1
save_model_every_n_iter = 15000
path_to_model = 'models/ladder'

cl = Ladder(input_shape=[32,32,3], n_classes=10, encoder_structure=param.cifar10_encoder,
    decoder_structure=param.cifar10_decoder, layer_importants=param.cifar10_layer_importants,
    noise_std=noise_std, do_train=True, scope='ladder')
cl.train_model(image_provider, labeled_data_loader, test_data_loader,
    batch_size, weight_decay, learn_rate_start, learn_rate_end, keep_prob,
    n_iter, save_model_every_n_iter, path_to_model)
cl.sess.close()
tf.reset_default_graph()
