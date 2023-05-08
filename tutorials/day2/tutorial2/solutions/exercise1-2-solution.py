"""
Read TFRecords from simple image datasets.
"""

from pathlib import Path
import sys

import tensorflow as tf


def _parse_example(example_proto):
    """Return the serialized `example_proto` as a tuple of the image data and
    its label.
    """
    # Task 2: Here, the `tf.train.Example` we created before is parsed.
    #
    #         `tf.io.parse_single_example` deserializes a Protobuf
    #         example, and extracts data based on a feature
    #         description into a Python `dict`.
    #
    #         Again, the provided implementation is for image data
    #         only, please extend it for the labels.

    # Step 1: Feature description.
    #         Please add the label to the `feature_description`.
    feature_description = {
        'image_data': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }

    # Step 2: Parsing based on the feature description.
    example = tf.io.parse_single_example(example_proto, feature_description)

    # Step 3: The tensor has to be deserialized because we serialized
    #         it as a bytestring before.
    #         This is not required for the label because it was stored
    #         as is; you can simply assign the label to a variable
    #         called `label`.
    image_data = tf.io.parse_tensor(example['image_data'], tf.uint8)
    label = example['label']
    return (image_data, label)


def from_tfrecords(path):
    """Return the TFRecords in `path` as a `tf.data.Dataset`."""
    if isinstance(path, Path):
        path = str(path.expanduser())

    raw_dataset = tf.data.TFRecordDataset([path])
    return raw_dataset.map(_parse_example)


# Script helpers

def _list_get(seq, index, default):
    """Return the value of `seq` at `index` or `default`.
    Like `dict.get`.
    """
    return seq[index] if len(seq) > index else default


def main():
    train_path = Path(
        _list_get(sys.argv, 1, './cifar10-train.tfrecords'),
    )
    train = from_tfrecords(train_path)

    print('First 2 items in train data:')
    for data in train.take(2):
        print(data)


if __name__ == '__main__':
    main()
