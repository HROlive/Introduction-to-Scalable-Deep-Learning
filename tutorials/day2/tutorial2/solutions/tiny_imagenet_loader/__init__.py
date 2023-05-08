from pathlib import Path
import sys

import tensorflow as tf

from .tiny_imagenet_loader import TinyImageNetLoader

# Reading

FEATURE_DESCRIPTION = {
    'label': tf.io.FixedLenFeature([], tf.int64),
    'image_data': tf.io.FixedLenFeature([], tf.string),
}


def _parse_example(example_proto):
    """Return the serialized `example_proto` as a tuple of the image data and
    its label.
    """
    example = tf.io.parse_single_example(example_proto, FEATURE_DESCRIPTION)
    # Convert the serialized tensor back
    image_data = tf.io.parse_tensor(example['image_data'], tf.uint8)
    return image_data, tf.cast(example['label'], tf.int32)


def from_tfrecords(path):
    """Return the TFRecords in `path` as a `tf.data.Dataset`."""
    if isinstance(path, Path):
        path = str(path.expanduser())

    raw_dataset = tf.data.TFRecordDataset([path])
    return raw_dataset.map(_parse_example)


# Writing

def _bytes_feature(value):
    """Return `value` as a `tf.train.BytesList`."""
    if isinstance(value, type(tf.constant(0))):
        # BytesList won't unpack a string from an EagerTensor.
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    """Return `value` as a `tf.train.Int64List`."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _to_tfexample(image_data, label):
    """Convert the given `image_data` and `label` to a `tf.train.Example`."""
    feature = {
        'label': _int64_feature(label.numpy().item()),
        'image_data': _bytes_feature(tf.io.serialize_tensor(image_data)),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def to_tfrecords(dataset, path):
    """Convert the given `dataset` to a TFRecords file stored at `path`."""
    if isinstance(path, Path):
        path = str(path.expanduser())

    with tf.io.TFRecordWriter(path) as writer:
        for (image, label) in dataset:
            tfexample = _to_tfexample(image, label)
            writer.write(tfexample.SerializeToString())


def _list_get(seq, index, default):
    """Return the value of `seq` at `index` or `default`.
    Like `dict.get`.
    """
    return seq[index] if len(seq) > index else default


def main():
    dataset_path = Path(_list_get(
        sys.argv,
        1,
        '/p/largedata/datasets/tiny-imagenet-200.zip',
    ))
    train_path = Path(
        _list_get(sys.argv, 2, './tiny-imagenet-200-train.tfrecords'),
    )
    val_path = Path(
        _list_get(sys.argv, 3, './tiny-imagenet-200-val.tfrecords'),
    )
    test_path = Path(
        _list_get(sys.argv, 4, './tiny-imagenet-200-test.tfrecords'),
    )

    loader = TinyImageNetLoader(dataset_path.expanduser())
    (train, val, test) = loader.load_datasets()
    to_tfrecords(train, train_path)
    to_tfrecords(val, val_path)
    to_tfrecords(test, test_path)


if __name__ == '__main__':
    main()
