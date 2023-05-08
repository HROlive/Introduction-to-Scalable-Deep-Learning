#!/usr/bin/python
__author__ = 'Jenia Jitsev, Mehdi Cherti'

# importing basic libraries and tools
import time
import os
import numpy as np
import argparse
from datetime import datetime

# Plotting and visualization
import pandas as pd
import matplotlib.pyplot as plt

# importing TensorFlow and Keras backbone
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers

# -------------- argument parser -------------------------------------

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--disable_gpu', '-no_gpu', dest='disable_gpu',
                        default=False, action='store_true', help="disable GPU usage")
    parser.add_argument('--learning_rate', '-lr',
                        help="SGD learning rate", type=float, default=0.1)
    parser.add_argument('--steps-per-epoch', help="steps per epoch", type=int, default=500)
    parser.add_argument('--epochs', '-ep',
                        help="number of epochs", type=int, default=1)
    parser.add_argument('--batch_size', '-b',
                        help="local batch size for a worker", type=int, default=32)
    args = parser.parse_args()
    return args

# ---------- dataset related -----------------------------

IMAGE_DIM = 64 # Image Dimension for the corresponding Dataset
CHANNEL_DIM = 3 # For standard color RGB images, 3 channels
INPUT_SHAPE = (IMAGE_DIM, IMAGE_DIM, CHANNEL_DIM)
NUM_CLASSES = 201

def decode(serialized_example):
    """Parses an image and label from the given `serialized_example`."""
    features = tf.io.parse_single_example(
        serialized_example,
        features={
            'image_data': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64),
        })
    label = tf.cast(features['label'], tf.int32)
    image = tf.io.parse_tensor(features['image_data'], tf.uint8)
    return image, label


def preprocess_rescale(image, label):
    """Rescale input from [0, 255] -> [0, 1] floats."""
    image = tf.cast(image, tf.float32) * (1. / 255)
    return image, label


# -------------- dataset handling ----------------------------

def get_dataset(filename, batch_size, epochs, training=True):
    dataset = tf.data.TFRecordDataset(filename)

    if training:
        # TODO Horovod: use shard method to get a portion of the data to the worker
        # using Horovod world size and worker's rank
        # Each horovod process will use an independent part of the dataset.
        # World size and global rank are used for partitioning data into subsets
        dataset = dataset.shard(hvd.size(), hvd.rank())

    dataset = dataset.map(decode, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(preprocess_rescale, num_parallel_calls=tf.data.AUTOTUNE)
    # Caching
    # For small datasets (e.g. MNIST, CIFAR,...), reading from .tfrecord can add significant overhead.
    # As those datasets fit in memory, it is possible to significantly improve the performance by caching or pre-loading the dataset.
    # https://www.tensorflow.org/datasets/performances#caching_the_dataset
    dataset = dataset.cache()
    if training:
        dataset_size = dataset.reduce(np.int64(0), lambda x, _: x + 1)
        dataset = dataset.shuffle(dataset_size)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

# --------------------------- MAIN BODY -----------------------------------

# TODO Horovod: Importing Horovod for Keras
import horovod.tensorflow.keras as hvd


# TODO Horovod: Step 2: initializing Horovod at the beginning of main training body
hvd.init()

# Set up GPU
gpus = tf.config.list_physical_devices('GPU')
# Currently, memory growth needs to be the same across GPUs.
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    # TODO Horovod: assign a GPU for the worker based on its local rank
    tf.config.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

args = arg_parse()

# ------------- Training Hyperparameters ----------------------
learning_rate = args.learning_rate
# ---- size of the local reference mini batch for the worker
batch_size = args.batch_size
# Number of epochs: one epoch is going through the whole dataset
epochs = args.epochs

print("Local batch size:", batch_size)
print("Number of GPUs:", hvd.size())
print("Effective batch size:", batch_size * hvd.size())

# Instantiate the model
model = tf.keras.applications.ResNet50(classes=NUM_CLASSES, weights=None, input_shape=INPUT_SHAPE)

optimizer = optimizers.SGD(lr=learning_rate, nesterov=True, momentum=0.9)

# TODO Horovod: Wrap the optimizer for Horovod
# wrap the optimizer using horovod distributed optimizer.
# The wrapped optimizer will aggregate the gradients from all
# workers and do a weight parameter update. This will make sure all the
# weights on all GPUs are synchronized with each other when performing an update step.
compression = hvd.Compression.fp16
optimizer = hvd.DistributedOptimizer(optimizer, compression=compression)

model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)

# TODO Horovod: output network model structure
# Let only worker with rank = 0 take care of this
if hvd.rank() == 0:
    model.summary()

# prepare TFRecordDataset data objects

train_data = get_dataset('/p/project/training2306/datasets/tiny-imagenet-200-train.tfrecords', batch_size=batch_size, epochs=epochs)
validation_data = get_dataset('/p/project/training2306/datasets/tiny-imagenet-200-val.tfrecords', batch_size=batch_size, epochs=1, training=False)
test_data = get_dataset('/p/project/training2306/datasets/tiny-imagenet-200-test.tfrecords', batch_size=batch_size, epochs=1, training=False)

callbacks = [
    # TODO Horovod: Introduce Horovod callbacks
    # 1. Broadcast the initial weights from rank zero so that all GPUs
    # have the exact same weights at the beginning.
    # 2. Gather metrics across the workers to report the average,
    # for instance for reporting validation accuracy
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),
    hvd.callbacks.MetricAverageCallback(),
]

# TODO Horovod: use verbose flag to provide it to the fit method such that only
# one worker with rank 0 will produce readable output
# Output produced only by the main worker with rank 0
verbose = 2 if (hvd.rank() == 0) else 0


# TODO Horovod: start timer to measure total training time
# Let only worker with rank = 0 take care of this
if hvd.rank() == 0:
    start = time.time()

history = model.fit(
    train_data,
    epochs=epochs,
    callbacks=callbacks,
    steps_per_epoch=args.steps_per_epoch,
    verbose=verbose
)

# TODO Horovod: measure total training time and print it.
# Let only worker with rank = 0 take care of this
if hvd.rank() == 0:
    nb_images_processed = args.steps_per_epoch * epochs * hvd.size() * batch_size
    duration = time.time() - start
    imsec = nb_images_processed / duration
    print(f"total training time in sec: {duration}")
    print(f"total images: {nb_images_processed}")
    print(f"total images/sec/gpu: {imsec/hvd.size()}")
    print(f"total images/sec:{imsec}")
