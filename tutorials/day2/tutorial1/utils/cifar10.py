"""
Load a stored CIFAR-10 dataset.

Code adapted from
https://github.com/tensorflow/tensorflow/blob/85c8b2a817f95a3e979ecd1ed95bff1dc1335cff/tensorflow/python/keras/datasets/cifar10.py.
"""

import os

import numpy as np
import tensorflow.python.keras.backend as K
from tensorflow.python.keras.datasets.cifar import load_batch


def load_data(path):
    """Load the [CIFAR-10
    dataset](https://www.cs.toronto.edu/~kriz/cifar.html).

    This is a dataset of 50,000 32×32 color training images and 10,000 test
    images, labeled over 10 categories. See more info at the
    [CIFAR homepage](https://www.cs.toronto.edu/~kriz/cifar.html).

    Arguments:
      path: Name of the directory containing the CIFAR-10 batch files.

    Returns:
      Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
      **x_train, x_test**: uint8 arrays of RGB image data with shape
        `(num_samples, 3, 32, 32)` if `tf.keras.backend.image_data_format()` is
        `'channels_first'`, or `(num_samples, 32, 32, 3)` if the data format
        is `'channels_last'`.
      **y_train, y_test**: uint8 arrays of category labels
        (integers in range 0-9) each with shape (num_samples, 1).

    """
    num_train_samples = 50000

    x_train = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
    y_train = np.empty((num_train_samples,), dtype='uint8')

    for i in range(1, 6):
        fpath = os.path.join(path, 'data_batch_' + str(i))
        (x_train[(i - 1) * 10000:i * 10000, :, :, :],
         y_train[(i - 1) * 10000:i * 10000]) = load_batch(fpath)

    fpath = os.path.join(path, 'test_batch')
    x_test, y_test = load_batch(fpath)

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    if K.image_data_format() == 'channels_last':
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)

    x_test = x_test.astype(x_train.dtype)
    y_test = y_test.astype(y_train.dtype)

    return (x_train, y_train), (x_test, y_test)
