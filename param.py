from collections import namedtuple
import tensorflow as tf

ConvoLayer = namedtuple('ConvoLayer', ['filters', 'kernel_size', 'strides',
    'padding', 'activation'])

MaxPool = namedtuple('MaxPool', ['pool_size', 'strides', 'padding'])

AvrPool = namedtuple('AvrPool', ['pool_size', 'strides', 'padding'])

DeConvoLayer = namedtuple('DeConvoLayer', ['filters', 'kernel_size', 'strides',
    'padding', 'activation'])

DenseLayer = namedtuple('DenseLayer', ['units', 'activation'])


# encoder = [                                                                                 # b x 28 x 28 x 1
#     ConvoLayer(filters=32, kernel_size=5, strides=1, padding='same', activation=tf.nn.relu), # b x 28 x 28 x 32
#     MaxPool(pool_size=2,                  strides=2, padding='valid'),                        # b x 14 x 14 x 32
#     ConvoLayer(filters=64, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu), # b x 14 x 14 x 64
#     ConvoLayer(filters=64, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu), # b x 14 x 14 x 64
#     MaxPool(pool_size=2,                  strides=2, padding='valid'),                        # b x 7 x 7 x 64
#     ConvoLayer(filters=128, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu),# b x 7 x 7 x 128
#     ConvoLayer(filters=10, kernel_size=1, strides=1, padding='same', activation=tf.nn.relu), # b x 7 x 7 x 10
#     AvrPool(pool_size=7,                  strides=1, padding='valid')                        # b x 1 x 1 10
#     ]

# decoder = [                                                                                     # b x 1 x 1 10
#     DeConvoLayer(filters=10, kernel_size=2, strides=7, padding='same', activation=None),  # b x 7 x 7 x 10
#     ConvoLayer(filters=128, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu),   # b x 7 x 7 x 128
#     ConvoLayer(filters=64, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu),    # b x 7 x 7 x 64
#     DeConvoLayer(filters=64, kernel_size=2, strides=2, padding='same', activation=None), # b x 14 x 14 x 64
#     ConvoLayer(filters=64, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu),    # b x 14 x 14 x 64
#     ConvoLayer(filters=32, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu),    # b x 14 x 14 x 32
#     DeConvoLayer(filters=32, kernel_size=2, strides=2, padding='same', activation=None), # b x 28 x 28 x 32
#     ConvoLayer(filters=1, kernel_size=5, strides=1, padding='same', activation=tf.nn.relu)      # b x 28 x 28 x 1
#     ]

encoder = [                                                                                 # b x 28 x 28 x 1
    DenseLayer(1000, tf.nn.relu),
    DenseLayer(500, tf.nn.relu),
    DenseLayer(100, tf.nn.relu),
    DenseLayer(10, tf.nn.softmax)
    ]

decoder = [                                                                                     # b x 1 x 1 10
    DenseLayer(100, tf.nn.relu),
    DenseLayer(500, tf.nn.relu),
    DenseLayer(1000, tf.nn.relu),
    DenseLayer(784, tf.nn.relu)
    ]