"""
DCGAN Horovod TensorFlow 2 training script, based on
https://www.tensorflow.org/tutorials/generative/dcgan and 
https://www.kaggle.com/laszlofazekas/cifar10-dcgan-example
"""

import argparse
from pathlib import Path
import time

import matplotlib
matplotlib.use('pdf')  # noqa
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
import numpy as np

# TODO Horovod: Importing Horovod for TensorFlow
...

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
# TODO Horovod: assign a GPU for the worker based on its local rank
if gpus:
    ...


def arg_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dry-run',
        dest='dry_run',
        action='store_true',
        help='Only one param update for sanity check',
    )
    parser.add_argument(
        '--epochs',
        '-ep',
        help='number of epochs',
        type=int,
        default=1,
    )
    parser.add_argument(
        '--batch_size',
        '-b',
        help='batch size',
        type=int,
        default=32,
    )
    args = parser.parse_args()
    return args


# define the standalone generator model
def make_generator_model(latent_dim):
    model = Sequential()
    # foundation for 4x4 image
    n_nodes = 256 * 4 * 4
    model.add(Dense(n_nodes, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((4, 4, 256)))
    # upsample to 8x8
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # upsample to 16x16
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # upsample to 32x32
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # output layer
    model.add(Conv2D(3, (3, 3), activation='tanh', padding='same'))
    return model


def make_discriminator_model(in_shape=(32, 32, 3)):
    model = Sequential()
    # normal
    model.add(Conv2D(64, (3, 3), padding='same', input_shape=in_shape))
    model.add(LeakyReLU(alpha=0.2))
    # downsample
    model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # downsample
    model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # downsample
    model.add(Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # classifier
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))
    return model


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


# Notice the use of `@tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    noise = tf.random.normal([images.shape[0], generator_input_size])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
    # TODO Horovod: wrap gen_tape with hvd.DistributedGradientTape
    # follow [Horovod's TF2 example](https://github.com/horovod/horovod/blob/master/examples/tensorflow2/tensorflow2_synthetic_benchmark.py#L81)
    gen_tape = ...
    # TODO Horovod: wrap disc_tape with hvd.DistributedGradientTape
    # follow [Horovod's TF2 example](https://github.com/horovod/horovod/blob/master/examples/tensorflow2/tensorflow2_synthetic_benchmark.py#L81)
    disc_tape = ...
    gradients_of_generator = gen_tape.gradient(gen_loss,
                                               generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(
        disc_loss, discriminator.trainable_variables
    )

    generator_optimizer.apply_gradients(
        zip(gradients_of_generator, generator.trainable_variables)
    )
    discriminator_optimizer.apply_gradients(
        zip(gradients_of_discriminator, discriminator.trainable_variables)
    )

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
    """Rescale input from [0, 255] -> [-1, 1] floats."""
    image = (tf.cast(image, tf.float32) - 127.5) / 127.5
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
    return dataset


def main(args):
    train_dataset = get_dataset('/p/project/training2306/datasets/train_CIFAR10.tfrecords', batch_size=args.batch_size, training=True)
    checkpoint_dir = Path('./training_checkpoints')
    checkpoint_prefix = checkpoint_dir / 'ckpt'
    checkpoint = tf.train.Checkpoint(
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        generator=generator,
        discriminator=discriminator,
    )

    num_examples_to_generate = 16

    # We will reuse this noise overtime (so it's easier
    # to visualize progress in the animated GIF).
    fixed_noise = tf.random.normal(
        [num_examples_to_generate, generator_input_size],
    )

    num_batches_processed = 0
    for epoch in range(args.epochs):
        start = time.time()
        for image_batch, _ in train_dataset:
            train_step(image_batch)
            num_batches_processed += 1

            # Broadcast after first gradient step to ensure
            # optimizer initialization.
            if num_batches_processed == 1:
                # TODO Horovod: broadcast generator variables 
                hvd.broadcast_variables(generator.variables, root_rank=...)
                # TODO Horovod: broadcast optimizer variables 
                hvd.broadcast_variables(generator_optimizer.variables(),
                                        root_rank=...)
                # TODO Horovod: broadcast discriminator variables 
                hvd.broadcast_variables(..., root_rank=...)
                # TODO Horovod: broadcast discriminator optimizer variables 
                hvd.broadcast_variables(...,
                                        root_rank=...)

            if args.dry_run:
                return

        # TODO Horovod: only render images for root rank
        if hvd.rank() == ...:
            generate_and_save_images(generator, epoch + 1, fixed_noise)
            checkpoint.save(file_prefix=str(checkpoint_prefix.expanduser()))
            print('Time for epoch {} is {} sec'.format(
                epoch + 1,
                time.time() - start,
            ))

    # Generate after the final epoch
    if hvd.rank() == 0:
        generate_and_save_images(generator, args.epochs, fixed_noise)


def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to `False`.
    # This is so all layers run in inference mode
    # which is relevant for batch normalization.
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow((predictions[i, :, :] * 127.5 + 127.5).numpy().astype("uint8"))
        plt.axis('off')
    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.close(fig)


if __name__ == '__main__':
    args = arg_parse()

    generator_input_size = 100
    generator = make_generator_model(generator_input_size)
    discriminator = make_discriminator_model()

    cross_entropy = tf.keras.losses.BinaryCrossentropy()

    generator_optimizer = tf.keras.optimizers.Adam(0.0002, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(0.0002, beta_1=0.5)

    main(args)
