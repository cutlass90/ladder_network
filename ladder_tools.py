from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.contrib.learn.python.learn.datasets.mnist import extract_images
from tensorflow.contrib.learn.python.learn.datasets.mnist import extract_labels
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet
import numpy as np

import tools


def read_data_sets(train_dir):
    # CVDF mirror of http://yann.lecun.com/exdb/mnist/
    SOURCE_URL = 'https://storage.googleapis.com/cvdf-datasets/mnist/'

    TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
    TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
    TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
    TEST_LABELS = 't10k-labels-idx1-ubyte.gz'

    local_file = base.maybe_download(TRAIN_IMAGES, train_dir,
                                   SOURCE_URL + TRAIN_IMAGES)
    with open(local_file, 'rb') as f:
        train_images = extract_images(f)

    local_file = base.maybe_download(TRAIN_LABELS, train_dir,
                                   SOURCE_URL + TRAIN_LABELS)
    with open(local_file, 'rb') as f:
        train_labels = extract_labels(f, one_hot=True)

    local_file = base.maybe_download(TEST_IMAGES, train_dir,
                                   SOURCE_URL + TEST_IMAGES)
    with open(local_file, 'rb') as f:
        test_images = extract_images(f)

    local_file = base.maybe_download(TEST_LABELS, train_dir,
                                   SOURCE_URL + TEST_LABELS)
    with open(local_file, 'rb') as f:
        test_labels = extract_labels(f, one_hot=True)

    return train_images, train_labels, test_images, test_labels


class ImageProvider:
    def __init__(self, images, labels):
        self.loader = DataSet(images, labels, one_hot=True)
    def next_batch(self, batch_size):
        return self.loader.next_batch(batch_size)[0]

def get_mnist_loaders(class_distrib_lab, class_distrib_unlab):
    train_im, train_l, test_im, test_l = read_data_sets("datasets/MNIST")
    lab_im, lab_t, rest_im, rest_t = tools.split_data_set(class_distrib_lab,
        train_im, train_l, verbose=True)
    unlab_im, unlab_t, rest_im, rest_t = tools.split_data_set(class_distrib_unlab,
        rest_im, rest_t)

    print('\n labeled summary:')
    tools.print_stat(lab_t)
    print('\n unlabeled summary:')
    tools.print_stat(unlab_t)

    labeled_data_loader = DataSet(lab_im, lab_t, one_hot=True)
    test_data_loader = DataSet(test_im, test_l, one_hot=True)

    image_provider = ImageProvider(np.vstack((lab_im, unlab_im)),
                                    np.vstack((lab_t, unlab_t)))

    return image_provider, labeled_data_loader, test_data_loader
