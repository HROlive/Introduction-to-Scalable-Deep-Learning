#!/usr/bin/python
__author__ = 'Jenia Jitsev, Mehdi Cherti'

# importing basic libraries and tools
import os
import numpy as np
import argparse
from functools import partial
import tensorflow as tf

import pandas as pd
import matplotlib
matplotlib.use('pdf')  # noqa
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Input, Activation, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras import optimizers

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import LearningRateScheduler

from datetime import datetime
from tensorflow import keras

import horovod.tensorflow.keras as hvd

from clr import OneCycleLR
from ResNet import ResNet56, ResNet32 #CIFAR-10 ResNet architectures following (He et al.)

from lamb import LAMB
from lars import LARSOptimizer


def device_init(args):
    """ Horovod init, GPUs config and stats """

    hvd.init()

    enable_gpu = not (args.disable_gpu)

    if enable_gpu:
        gpus = tf.config.list_physical_devices('GPU')
        gpu_id = hvd.local_rank()

        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            if gpus:
                tf.config.set_visible_devices(gpus[gpu_id], 'GPU')

            logical_gpus = tf.config.list_logical_devices('GPU')
            # Diagnostic print info
            # print(len(logical_gpus), "Visible Logical GPU, GPU ID - ", gpus[gpu_id])
        except RuntimeError as e:
            # GPU conf must be set before GPUs have been initialized
            print(e)
    else:
        print("*** WARNING ! ***, GPUs will be disabled !")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--disable_gpu', '-no_gpu', dest='disable_gpu', default=False, action='store_true', help="disable GPU usage")
    parser.add_argument('--learning_rate', '-lr', help="reference SGD learning rate (for single GPU training)", type=float, default=0.1)
    parser.add_argument('--enable_lr_rescaling', dest='enable_lr_rescaling', default=False, action='store_true', help="enable learning rate rescaling for distributed training")
    parser.add_argument('--epochs', '-ep', help="number of epochs", type=int, default=150)
    parser.add_argument('--lr_scheduler', help="Learning rate scheduler. Options are: none/step_wise_decay/step_wise_decay_with_warmup/clr", type=str, default="none")
    parser.add_argument('--optimizer', help="Optimizer to use. Options are: SGD/LARS/LAMB", type=str, default="SGD")
    parser.add_argument('--batch_size', '-b', help="batch size", type=int, default=32)
    parser.add_argument('--disable_input_rescale', dest='norm_rescaling', default=False, action='store_true', help="disable input rescaling")
    parser.add_argument('--plot_filename', '-pf', help="plot file name", type=str, default='plot_accuracy')
    parser.add_argument("--weight_decay", help="Weight decay", type=float, default=0.0001)
    parser.add_argument("--result_file", help="Results CSV file path, will contain learning curves of train/validation", type=str, default="results.csv")
    args=parser.parse_args()
    return args

def plot_accuracy(history, plot_filename):

    filename = plot_filename

    fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True, figsize=(12, 6))
    history_df = pd.DataFrame(history.history)
    history_df["epoch"] = history.epoch

    # For TF 1.x 'acc' and 'val_acc'
    history_df.plot(x="epoch", y=["loss", "val_loss"], ax=ax0)
    history_df.plot(x="epoch", y=["accuracy", "val_accuracy"], ax=ax1);

    fig.savefig(filename + ".pdf")


def decode(serialized_example):
    """Parses an image and label from the given `serialized_example`."""
    features = tf.io.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64),
        })
    label = tf.cast(features['label'], tf.int32)
    image = tf.io.decode_raw(features['image_raw'], tf.uint8)
    image = tf.reshape(image, (32, 32, 3))
    return image, label

def preprocess_rescale(image, label):
    """Rescale input from [0, 255] -> [0, 1] floats."""
    image = tf.cast(image, tf.float32) * (1. / 255)
    return image, label

def get_dataset(filename, batch_size, training=True):
    dataset = tf.data.TFRecordDataset(filename)
    # shard the dataset, each horovod process will use an independent
    # part of the dataset.
    if training:
        dataset = dataset.shard(hvd.size(), hvd.rank())
    dataset = dataset.map(decode, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(preprocess_rescale, num_parallel_calls=tf.data.AUTOTUNE)

    # Caching
    # For small datasets (e.g. MNIST, CIFAR,...), reading from .tfrecord can add significant overhead.
    # As those datasets fit in memory, it is possible to significantly improve the performance by caching or pre-loading the dataset.
    # https://www.tensorflow.org/datasets/performances#caching_the_dataset
    dataset = dataset.cache()
    
    if training:
        # shuffle only during training, during validation order does not matter
        # HOROVOD related: shuffling on portion of the dataset that is assigned to a worker
        dataset_size = dataset.reduce(np.int64(0), lambda x, _: x + 1)
        dataset = dataset.shuffle(dataset_size)
        dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)

    # Most dataset input pipelines should end with a call to prefetch.
    # This allows later elements to be prepared while the current element is being processed.
    # This often improves latency and throughput, at the cost of using additional memory to store prefetched elements.
    #
    # https://www.tensorflow.org/guide/data_performance#prefetching
    #
    # Prefetching overlaps the preprocessing and model execution of a training step.
    # While the model is executing training step s, the input pipeline is reading the data for step s+1.
    # Doing so reduces the step time to the maximum (as opposed to the sum) of the training and the time it takes to extract the data.
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    # dataset = dataset.repeat()
    return dataset

def augment(image, label, pad_size=4, crop_size=32):
    # CIFAR-10 augmentation following https://arxiv.org/abs/1512.03385
    image = tf.pad(image, [[pad_size, pad_size], [pad_size, pad_size], [0, 0]])
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_crop(image, [crop_size, crop_size, 3])
    return image, label

 
def step_wise_decay(epoch, lr, init_lr=0.1):
    # divise learning rate by 10 at epoch 80 and at epoch 120
    if epoch < 80:
        return init_lr
    elif epoch < 120:
        return init_lr / 10.
    else:
        return init_lr / 100.

def step_wise_decay_with_warmup(epoch, lr, init_lr=0.1, warmup_epochs=5):
    # - warmup for the first 5 epochs
    # - divise learning rate by 10 at epoch 80 and at epoch 120
    if epoch < 5:
        # Scale linearly from (init_lr/hvd.size()) to (init_lr) in the
        # first five epochs.
        # As we use the linear scaling rule, we multiply further the learning rate
        # by hvd.size(). Thus, this will result in starting from `init_lr` and linearly
        # increasing to `init_lr * hvd.size()`, which is the desired behavior, following Goyal et al. (https://arxiv.org/abs/1706.02677).
        return (init_lr/hvd.size())*(((hvd.size()-1)/warmup_epochs)*epoch + 1)
    elif epoch < 80:
        return init_lr
    elif epoch < 120:
        return init_lr / 10.
    else:
        return init_lr / 100.

def main(args):

    # filename for plotting
    plot_filename = args.plot_filename

    # Version Info
    print("TensorFlow version in use: ", tf.version.VERSION)
    print("Keras version in use: ", tf.keras.__version__)

    SEED_NUM = 16823
    np.random.seed(SEED_NUM)  # for reproducibility

    num_classes = 10

    # this will determine the stability and speed of convergence during learning
    # learning rate too low - very slow convergence, can stall learning entirely
    # learning rate too high - instable, may diverge so that learning collapses entirely
    # ** ATTENTION **: For distributed training, learning rate reference for single GPU
    # usually has to be rescaled to accomodate increasing effective batch size when using
    # data parallel training with multiple GPUs
    learning_rate = args.learning_rate

    # ---- size of the mini batch
    # too small: computationally inefficient, no advantage through vectorization; slower convergence via high noise
    # too high: lack of noise, stochasticity in updates; danger of convergence to bad (flat) local minima
    batch_size = args.batch_size # 32; 128; 256

    # Number of epochs: one epoch is going through the whole dataset
    epochs = args.epochs
    
    # Learning rate linear rescaling for distributed training: enabled by default
    if args.enable_lr_rescaling:
        # rescaling learning rate by the number of workers / GPUs (1 worker per GPU)
        learning_rate = learning_rate * hvd.size()
    else:
        print('*** WARNING ! *** : learning rate will be NOT rescaled for distributed training!')

    # CIFAR=10 input dimensions : width, height, channels
    input_shape = (32, 32, 3)
    input_channels = input_shape[2]
    inputs = Input(shape=input_shape)

    # for LAMB and LARS, the weight decay is integrated into the optimizer (see below)
    model = ResNet56(num_classes, input_shape, weight_decay=args.weight_decay if args.optimizer == "SGD" else 0.)
    
    # wrap the optimizer using horovod distributed optimizer.
    # The wrapped optimizer will aggregate the gradients from all
    # gpus and do a parameter update. This will make sure all the
    # weights on all GPUs are synchronized with each other.
    # learning rate is scaled linearly by the number of GPUs to cope
    # with the increased **effective batch size**, see
    # "Accurate, Large Minibatch SGD:Training ImageNet in 1 Hour"
    # for more details.
    # learning rate scaling already performed based on the flag above

    if args.optimizer == "SGD":
        optimizer = optimizers.SGD(lr=learning_rate, nesterov=True, momentum=0.9)
    elif args.optimizer == "LARS":
        optimizer = LARSOptimizer(learning_rate=learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == "LAMB":
        optimizer = LAMB(learning_rate=learning_rate, weight_decay_rate=args.weight_decay)
    else:
        raise ValueError("Unknown optimizer, optimizer should be SGD or LARS or LAMB")
    optimizer = hvd.DistributedOptimizer(optimizer)

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # print out network description
    # Output produced only by the main worker with rank 0
    if hvd.rank() == 0:
        model.summary()

    # prepare TFRecordDataset data objects
    train_data = get_dataset('/p/project/training2306/datasets/train_CIFAR10.tfrecords', batch_size=batch_size, training=True)
    validation_data = get_dataset('/p/project/training2306/datasets/valid_CIFAR10.tfrecords', batch_size=batch_size, training=False)
    test_data = get_dataset('/p/project/training2306/datasets/test_CIFAR10.tfrecords', batch_size=batch_size, training=False)
    callbacks = [
        # broadcast the initial weights from rank zero so that all GPUs
        # have the exact same weights at the beginning.
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),
        hvd.callbacks.MetricAverageCallback(),
    ]

    if args.lr_scheduler == "step_wise_decay":
        callbacks.append(LearningRateScheduler(partial(step_wise_decay, init_lr=learning_rate), verbose=1))
    elif args.lr_scheduler == "step_wise_decay_with_warmup":
        callbacks.append(LearningRateScheduler(partial(step_wise_decay_with_warmup, init_lr=learning_rate), verbose=1))
    elif args.lr_scheduler == "clr":
        callbacks.append(OneCycleLR(50000, batch_size * hvd.size(), max_lr=learning_rate))
    elif args.lr_scheduler == "none":
        pass
    else:
        raise ValueError(f"Unknown scheduler: {args.lr_scheduler}. Please choose between none or step_wise_decay or step_wise_decay_with_warmup or clr.")
    # perform training; dataset.repeat() NOT used - epochs in fit
    # Output produced only by the main worker with rank 0
    verbose = 1 if (hvd.rank() == 0) else 0
    history = model.fit(train_data, validation_data=validation_data, epochs=epochs, callbacks=callbacks, verbose=verbose)

    # plot training and validation loss and accuracy
    # Output produced only by the main worker with rank 0
    if hvd.rank() == 0:
        history_df = pd.DataFrame(history.history)
        history_df["epoch"] = history.epoch
        history_df.to_csv(args.result_file, index=False)
        show_accuracy(model, test_data)

def show_accuracy(model, test_data):
    Y_test = []
    for features, label in test_data:
        Y_test.append(label.numpy())
    Y_test = np.concatenate(Y_test, axis=0)
    # obtain probabilities across classes
    y_prob = model.predict(test_data)
    # select class with the highest probability as predicted label
    y_predicted = y_prob.argmax(axis=1)
    print("Test Accuracy: %0.4f" % np.mean(y_predicted == Y_test))

if __name__ == '__main__':
    args = arg_parse()
    device_init(args)
    main(args)
