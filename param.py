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


# encoder = [                                                                                     # b x 28 x 28 x 1
#     ConvoLayer(filters=1000, kernel_size=26, strides=1, padding='valid', activation=tf.nn.relu),# b x 3 x 3 x 1000
#     ConvoLayer(filters=500, kernel_size=1, strides=1, padding='valid', activation=tf.nn.relu),  # b x 3 x 3 x 500
#     ConvoLayer(filters=100, kernel_size=1, strides=1, padding='valid', activation=tf.nn.relu),  # b x 3 x 3 x 100
#     ConvoLayer(filters=10, kernel_size=1, strides=1, padding='valid', activation=tf.nn.relu),   # b x 3 x 3 x 10
#     AvrPool(pool_size=3, strides=1, padding='valid'),                                            # b x 1 x 1 x 10
#     Reshape([-1, 10]),                                                                           # b x 10
#     DenseLayer(10, tf.nn.softmax)                                                                # b x 10
#     ]

# decoder = [                                                                                     # b x 10
#     DenseLayer(10, tf.nn.relu),                                                                  # b x 10
#     Reshape([-1, 1, 1, 10]),                                                                    # b x 1 x 1 x 10
#     DeConvoLayer(filters=10, kernel_size=3, strides=1, padding='valid', activation=tf.nn.relu),  # b x 3 x 3 x 10
#     ConvoLayer(filters=100, kernel_size=1, strides=1, padding='valid', activation=tf.nn.relu),  # b x 3 x 3 x 100
#     ConvoLayer(filters=500, kernel_size=1, strides=1, padding='valid', activation=tf.nn.relu),  # b x 3 x 3 x 500
#     ConvoLayer(filters=1000, kernel_size=1, strides=1, padding='valid', activation=tf.nn.relu), # b x 3 x 3 x 1000
#     DeConvoLayer(filters=1, kernel_size=26, strides=1, padding='valid', activation=tf.nn.relu) # b x 28 x 28 x 1
#     ]

encoder = [                        # b x 784
    DenseLayer(1000, tf.nn.relu),  # b x 1000
    DenseLayer(500, tf.nn.relu),   # b x 500
    DenseLayer(100, tf.nn.relu),   # b x 100
    DenseLayer(10, tf.nn.softmax)  # b x 10
    ]

decoder = [                                                                                     # b x 1 x 1 10
    DenseLayer(100, tf.nn.relu),
    DenseLayer(500, tf.nn.relu),
    DenseLayer(1000, tf.nn.relu),
    DenseLayer(784, tf.nn.relu)
    ]