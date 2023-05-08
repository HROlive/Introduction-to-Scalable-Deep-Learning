from zipfile import ZipFile

import tensorflow as tf

from .tiny_imagenet_parser import TinyImageNetParser


class TinyImageNetLoader:
    """Load the Tiny-ImageNet-200 dataset as a `tf.data.Dataset`.

    If loading multiple datasets, with the `self.load_dataset` method, call
    `self.reset` to reset the internal state/cache.
    There are 201 classes in total; 200 in the dataset plus unknown classes in
    the test dataset.
    """

    def __init__(self, dataset_path=None):
        self._dataset_path = dataset_path
        self.reset()

    def reset(self):
        self._label_parser = None
        self._image_names = None
        self._num_classes = None

    def num_classes(self):
        """Return the number of classes in the dataset."""
        if self._num_classes is None:
            raise ValueError(
                'cannot output number of classes prior to loading',
            )
        return self._num_classes

    @staticmethod
    def _is_jpeg(name):
        """Return whether `name` corresponds to a JPEG image."""
        return name.endswith('.JPEG')

    @staticmethod
    def _in_dataset(name, dataset):
        """Return whether `name` is contained in `dataset`.
        `dataset` should be in `('train', 'val', 'test')`.
        """
        return name.startswith('tiny-imagenet-200/' + dataset + '/')

    @staticmethod
    def load_img(zipped_data, name):
        """Return an image with `name` loaded from the `zipped_data`."""
        with zipped_data.open(name) as image_file:
            image = tf.image.decode_jpeg(image_file.read(), channels=3)
            return image

    @staticmethod
    def label_img(label_parser, name):
        """Return the label for an image with `name` in the zipped data.
        `label_parser` is used to parse the label for the `name`."""
        class_id = label_parser.parse_class_id(name)
        label = label_parser.index_of_class_id(class_id)
        # Use tf.int32 so the data aligns better
        # (keeping bytes divisible by 4)
        return tf.constant(label, dtype=tf.int32)

    def load_dataset(self, zipped_data, dataset):
        """Load the specified `dataset` from `zipped_data`.

        `dataset` must be a string in `['train', 'val', 'test']`.
        If using this to load datasets in different `zipped_data`, call
        `self.reset` to properly parse the other data.
        """
        if dataset not in ['train', 'val', 'test']:
            raise ValueError(
                "please only give a dataset in `['train', 'val', 'test']`"
            )

        # Cache label parser
        if self._label_parser is None:
            self._label_parser = TinyImageNetParser(zipped_data)
            self._num_classes = self._label_parser.NUM_CLASSES

        # Cache image names
        if self._image_names is None:
            files = zipped_data.namelist()
            self._image_names = list(filter(self._is_jpeg, files))

        set_images = list(filter(
            lambda name: self._in_dataset(name, dataset),
            self._image_names,
        ))

        # The ``x``, or 'feature', part of the set
        image_data = list(map(
            lambda name: self.load_img(zipped_data, name),
            set_images,
        ))
        image_data = tf.data.Dataset.from_tensor_slices(image_data)

        # The ``y``, or 'target', part of the set
        labels = list(map(
            lambda name: self.label_img(self._label_parser, name),
            set_images,
        ))
        labels = tf.data.Dataset.from_tensor_slices(labels)

        combined_data = tf.data.Dataset.zip((image_data, labels))
        return combined_data

    def load_datasets(self, dataset_path=None):
        """Return the Tiny-ImageNet-200 dataset as a `list`
        of `tf.data.Dataset`s.

        The list contains the train, validation, and test datasets in
        that order.
        There are 201 classes in total (200 plus an unknown class).
        Labels 0–199 represent the 200 classes in the dataset while
        label 200 represents the unknown class.
        """
        if dataset_path is None and self._dataset_path:
            return self.load_datasets(self._dataset_path)

        self.reset()
        with ZipFile(self._dataset_path, 'r') as zipped_data:
            datasets = [self.load_dataset(zipped_data, dataset)
                        for dataset in ['train', 'val', 'test']]

        self.reset()
        return datasets
