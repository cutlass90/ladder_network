from collections import namedtuple
import tensorflow as tf

ConvoLayer = namedtuple('ConvoLayer', ['filters', 'kernel_size', 'strides',
    'padding', 'activation'])

MaxPool = namedtuple('MaxPool', ['pool_size', 'strides', 'padding'])

AvrPool = namedtuple('AvrPool', ['pool_size', 'strides', 'padding'])

DeConvoLayer = namedtuple('DeConvoLayer', ['filters', 'kernel_size', 'strides',
    'padding', 'activation'])

DenseLayer = namedtuple('DenseLayer', ['units', 'activation'])

Reshape = namedtuple('Reshape', ['shape'])


# ---=== convo model ===---
convo_encoder = [                                                                                    # b x 28 x 28 x 1
    Reshape([-1, 28,28,1]),
    ConvoLayer(filters=1000, kernel_size=26, strides=1, padding='valid', activation=tf.nn.relu),# b x 3 x 3 x 1000
    ConvoLayer(filters=500, kernel_size=1, strides=1, padding='valid', activation=tf.nn.relu),  # b x 3 x 3 x 500
    ConvoLayer(filters=100, kernel_size=1, strides=1, padding='valid', activation=tf.nn.relu),  # b x 3 x 3 x 100
    ConvoLayer(filters=10, kernel_size=1, strides=1, padding='valid', activation=tf.nn.relu),   # b x 3 x 3 x 10
    AvrPool(pool_size=3, strides=1, padding='valid'),                                            # b x 1 x 1 x 10
    Reshape([-1, 10]),                                                                           # b x 10
    DenseLayer(10, tf.nn.softmax)                                                                # b x 10
    ]

convo_decoder = [                                                                                     # b x 10
    DenseLayer(10, None),                                                                  # b x 10
    Reshape([-1, 1, 1, 10]),                                                                    # b x 1 x 1 x 10
    DeConvoLayer(filters=10, kernel_size=3, strides=1, padding='valid', activation=None),  # b x 3 x 3 x 10
    ConvoLayer(filters=100, kernel_size=1, strides=1, padding='valid', activation=None),  # b x 3 x 3 x 100
    ConvoLayer(filters=500, kernel_size=1, strides=1, padding='valid', activation=None),  # b x 3 x 3 x 500
    ConvoLayer(filters=1000, kernel_size=1, strides=1, padding='valid', activation=None), # b x 3 x 3 x 1000
    DeConvoLayer(filters=1, kernel_size=26, strides=1, padding='valid', activation=None) # b x 28 x 28 x 1
    ]
convo_layer_importants = [1000, 10, 0.1, 0.1, 0.1, 0.1, 0.1]




# ---=== CIFAR10 convo model ===---
cifar10_encoder = [                                                                                     # b x 32 x 32 x 3
    ConvoLayer(filters=96, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu),# b x 32 x 32 x 96
    ConvoLayer(filters=96, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu),# b x 32 x 32 x 96
    ConvoLayer(filters=96, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu),# b x 32 x 32 x 96
    MaxPool(pool_size=2, strides=2, padding='valid'),                                        # b x 16 x 16 x 96
    ConvoLayer(filters=192, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu),# b x 16 x 16 x 192
    ConvoLayer(filters=192, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu),# b x 16 x 16 x 192
    ConvoLayer(filters=192, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu),# b x 16 x 16 x 192
    MaxPool(pool_size=2, strides=2, padding='valid'),                                         # b x 8 x 8 x 192
    ConvoLayer(filters=192, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu),# b x 8 x 8 x 192
    ConvoLayer(filters=192, kernel_size=1, strides=1, padding='same', activation=tf.nn.relu),# b x 9 x 9 x 192
    ConvoLayer(filters=10, kernel_size=1, strides=1, padding='same', activation=tf.nn.relu),# b x 8 x 8 x 10
    AvrPool(pool_size=8, strides=1, padding='valid'),                                       # b x 1 x 1 x 10
    Reshape([-1, 10])                                                                        
    ]

cifar10_decoder = [                                                                                     # b x 10
    Reshape([-1, 1, 1, 10]),                                                               
    DeConvoLayer(filters=10, kernel_size=8, strides=1, padding='valid', activation=None),  # b x 8 x 8 x 10
    ConvoLayer(filters=192, kernel_size=1, strides=1, padding='same', activation=None),  # b x 8 x 8 x 192
    ConvoLayer(filters=192, kernel_size=1, strides=1, padding='same', activation=None),  # b x 8 x 8 x 192
    ConvoLayer(filters=192, kernel_size=3, strides=1, padding='same', activation=None),  # b x 8 x 8 x 192
    DeConvoLayer(filters=192, kernel_size=2, strides=2, padding='valid', activation=None),# b x 16 x 16 x 192
    ConvoLayer(filters=192, kernel_size=3, strides=1, padding='same', activation=None),  # b x 16 x 16 x 192
    ConvoLayer(filters=192, kernel_size=3, strides=1, padding='same', activation=None),  # b x 16 x 16 x 192
    ConvoLayer(filters=96, kernel_size=3, strides=1, padding='same', activation=None),  # b x 18 x 18 x 96
    DeConvoLayer(filters=96, kernel_size=2, strides=2, padding='valid', activation=None),# b x 32 x 32 x 96
    ConvoLayer(filters=96, kernel_size=3, strides=1, padding='same', activation=None),  # b x 32 x 32 x 96
    ConvoLayer(filters=96, kernel_size=3, strides=1, padding='same', activation=None),  # b x 32 x 32 x 96
    ConvoLayer(filters=3, kernel_size=3, strides=1, padding='same', activation=None),  # b x 32 x 32 x 3
    ]
cifar10_layer_importants = [1000, 10, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]



# ---=== dense model ===---
dense_encoder = [                        # b x 784
    DenseLayer(1000, tf.nn.relu),  # b x 1000
    DenseLayer(625, tf.nn.relu),   # b x 625
    DenseLayer(100, tf.nn.relu),   # b x 100
    DenseLayer(10, tf.nn.softmax)  # b x 10
    ]

dense_decoder = [                  # b x 10
    DenseLayer(100, None),   # b x 100
    DenseLayer(625, None),
    DenseLayer(1000, None),
    DenseLayer(784, None)
    ]
dense_layer_importants = [1000, 10,  0.1, 0.1, 0.1]


