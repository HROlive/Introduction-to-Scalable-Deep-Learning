"""
Write TFRecords from simple image datasets.
"""

import argparse
from pathlib import Path

import tensorflow as tf

import cifar10


# Conversion helpers for scalars

def _bytes_feature(value):
    """Return `value` as a `tf.train.BytesList` feature."""
    if isinstance(value, type(tf.constant(0))):
        # BytesList won't unpack a string from an EagerTensor.
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    """Return `value` as a `tf.train.Int64List` feature."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# Data conversion

def _to_tfexample(image_data, label):
    """Convert the given `image_data` and `label` to a `tf.train.Example`."""
    # Task 1: This function converts a pair of image (tf tensor)
    #         and label (scalar Int64) to a tf.train.Example.
    #
    #         This requires a few steps that are already implemented
    #         for image_data, but not for the label.
    #
    #         Extend the code for the label.

    # Convert Tensor to a stream of bytes
    image_data_as_bytes = tf.io.serialize_tensor(image_data)
    # The labels can be converted to int64 directly

    # Convert each label and image to a `tf.train.Feature`
    image_features = _bytes_feature(image_data_as_bytes)
    label_features = _int64_feature(label)

    # Combine in a dictionary and store as `tf.train.Features` which
    # is essentially a Protobuf `dict` mapping strings to individual
    # `tf.train.Feature`s (notice the subtle plural difference).
    feature_dict = {
                'image_data': image_features,
                'label': label_features,
    }

    # Create the Protobuf version of the feature_dict
    features = tf.train.Features(feature=feature_dict)

    # Finally, create a `tf.train.Example` out of it.
    # Our `tf.train.Example` only holds the `tf.train.Features`. It
    # may also contain other data, which we don't use.
    example = tf.train.Example(features=features)
    return example


def to_tfrecords(dataset, path):
    """Convert the given `dataset` to a TFRecords file stored at `path`."""
    if isinstance(path, Path):
        path = str(path.expanduser())

    (images, labels) = dataset
    with tf.io.TFRecordWriter(path) as writer:
        for (image, label) in zip(images, labels):
            # Return the scalar in the NumPy array using `item`.
            label = label.item()
            tfexample = _to_tfexample(image, label)
            writer.write(tfexample.SerializeToString())


# Script helpers

def _parse_args():
    """Return parsed command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'dataset_dir',
        nargs='?',
        default='/p/project/training2306/datasets/cifar10',
        help='Where the CIFAR-10 dataset batches are stored.',
    )
    parser.add_argument(
        'output_dir',
        nargs='?',
        default='.',
        help='Where to output the TFRecord files.',
    )
    return parser.parse_args()


def generate_tfrecords(dataset_dir, output_dir):
    """Generate TFRecord files at `output_dir` for the CIFAR-10 dataset
    whose batches are at `dataset_dir`.
    """
    dataset_dir = Path(dataset_dir).expanduser()
    output_dir = Path(output_dir).expanduser()

    train_path = output_dir / 'cifar10-train.tfrecords'
    test_path = output_dir / 'cifar10-test.tfrecords'

    (train, test) = cifar10.load_data(str(dataset_dir))
    to_tfrecords(train, train_path)
    to_tfrecords(test, test_path)


if __name__ == '__main__':
    args = _parse_args()
    generate_tfrecords(args.dataset_dir, args.output_dir)
