import tools
import ladder_tools
import param
from ladder import Ladder


# class_distrib_lab = {i:i+1 for i in range(10)}
# class_distrib_lab = {0:18, 1:16, 2:15, 3:13, 4:11, 5:9, 6:7, 7:5, 8:4, 9:2}
class_distrib_lab = {i:10 for i in range(10)}


# class_distrib_unlab = {i:5400 for i in range(10)}
# class_distrib_unlab = {0:5400, 1:4860, 2:4320, 3:3780, 4:3240, 5:2700, 6:2160,
#                  7:1620, 8:1080, 9:540}
class_distrib_unlab = {i:5400 for i in range(10)}
image_provider, labeled_data_loader, test_data_loader = ladder_tools.get_mnist_loaders(
    class_distrib_lab, class_distrib_unlab)


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




