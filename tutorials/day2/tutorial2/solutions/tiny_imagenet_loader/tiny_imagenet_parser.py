"""
Label parser for the Tiny-ImageNet-200 dataset.
Grabs both the class IDs and descriptions.

Currently only works in-memory for the zipped dataset.
"""

import numpy as np


def scalar_to_dtype(scalar, dtype):
    """Convert a scalar to a `np.dtype`."""
    # We cannot use the `item` method on a 0-D array because it gives a
    # different type.
    return np.array([scalar], dtype=dtype)[0]


def rstrip_newline(text):
    """Strip a single newline from the end of `text`."""
    if text[-1] == '\n':
        return text[:-1]
    return text


class TinyImageNetParser:
    """Label parser for the zipped Tiny-ImageNet-200 dataset."""

    # 200 in Tiny-ImageNet plus the 'unknown' class for the test set.
    NUM_CLASSES = 201
    """Number of total classes."""
    UNKNOWN_CLASS = 'unknown'
    """Label for the unknown class."""
    _MAX_CLASS_ID_LENGTH = 9
    """The maximum expected string length of a class ID."""
    _CLASS_ID_DTYPE = 'U' + str(_MAX_CLASS_ID_LENGTH)
    """`np.dtype` of class IDs for efficient storage."""

    def __init__(self, zip_file=None):
        """Construct a new `TinyImageNetParser` parsing the dataset in the
        **already opened** `zip_file`.
        """
        # Note: `class_words` do not uniquely identify a class
        self._class_words = np.empty(self.NUM_CLASSES, dtype=object)

        self._zip_file = zip_file
        if zip_file is not None:
            self.parse_classes(zip_file)

    def _require_zip_file_set(self, zip_file=None):
        """Raise an error if no zip file has been set yet.

        If `zip_file` is given, set it beforehand."""
        if zip_file is not None:
            self.set_zip_file(zip_file)
        if self._zip_file is None:
            raise ValueError('please set a zip file first')

    def parse_class_id(self, path):
        """Return a class ID label for the given image path.

        Raise an error if the path could not be parsed.
        """
        self._require_zip_file_set()

        # paths are of the form '[...]/{val,test}/images/file.JPEG'
        # or '[...]/train/class_id/images/file.JPEG'
        path_parts = path.split('/')
        if len(path_parts) >= 3:
            if path_parts[-3] == 'val':
                # Validation set
                return self._parse_validation_class(path_parts[-1])
            elif path_parts[-3] == 'test':
                # Test set
                return self.UNKNOWN_CLASS

            # Training set
            return path_parts[-3]

        # Something else entirely
        raise ValueError('unexpected image path')

    def index_of_class_id(self, class_id):
        """Return the index of `class_id`."""
        if class_id == self.UNKNOWN_CLASS:
            # Last index
            return self.NUM_CLASSES - 1

        pos = np.searchsorted(self._class_ids, class_id)
        if pos >= len(self._class_ids) or self._class_ids[pos] != class_id:
            raise ValueError('class ID {} not found'.format(class_id))
        return pos

    def class_id_to_description(self, class_id):
        """Return the description for `class_id`.

        Make sure you have parsed `self.parse_class_words` before
        trying to call this.
        """
        pos = self.index_of_class_id(self.to_class_id_dtype(class_id))
        return self._class_words[pos]

    def _get_validation_file_number(self, filename):
        """Return the number in the given validation filename.

        Expected to be only the filename, without its directory or
        other prefixes.
        """
        # Filenames are of the form val_X.JPEG,
        # where X is a number from 0 to 9999
        return int(filename[4:filename.find('.')])

    def _skip_val_class_ids_file_lines(
            self,
            val_class_ids_file,
            filename,
    ):
        """Skip lines in `val_class_ids_file` until we reach the one
        containing `filename` with another call to the function `next`.
        """
        filename_num = self._get_validation_file_number(filename)

        # File names are ordered, so skip until we are at the
        # correct line.
        for _ in range(filename_num):
            next(val_class_ids_file)

    def _parse_validation_class(self, filename):
        """Return the class ID for the validation image with `filename`."""
        with self._zip_file.open(
                'tiny-imagenet-200/val/val_annotations.txt'
        ) as val_class_ids_file:
            self._skip_val_class_ids_file_lines(val_class_ids_file, filename)
            line = next(val_class_ids_file)

            (line_filename, class_id) = line.split(b'\t')[:2]
            if line_filename.decode() != filename:
                raise ValueError(
                    'validation set filename mismatch ({} != {})'.format(
                        line_filename.decode(),
                        filename,
                    )
                )

            return class_id.decode()

    def set_zip_file(self, zip_file):
        """Set the internal zip file to `zip_file`."""
        self._zip_file = zip_file

    def parse_classes(self, zip_file=None):
        """Parse all class-relevant files in the internal zip file or
        `zip_file` if given.
        """
        self._require_zip_file_set(zip_file)

        self.parse_class_id_file()
        self.parse_class_words_file()

        return (self._class_ids, self._class_words)

    def parse_class_id_file(self, zip_file=None):
        """Parse the file containing the class IDs in the internal zip
        file or `zip_file` if given.
        """
        self._require_zip_file_set(zip_file)

        # Get IDs used
        with self._zip_file.open(
                'tiny-imagenet-200/wnids.txt'
        ) as class_ids_file:
            self.parse_class_ids(class_ids_file)

    def parse_class_ids(self, class_ids_file):
        """Parse all class IDs in `class_ids_file`."""
        lines = map(rstrip_newline, class_ids_file)
        # Unknown class is not included so subtract 1
        self._class_ids = np.fromiter(
            lines,
            dtype=self._CLASS_ID_DTYPE,
            count=self.NUM_CLASSES - 1,
        )
        # Class IDs do not come sorted
        self._class_ids.sort()

    def to_class_id_dtype(self, scalar):
        """Return `scalar` converted to the type class IDs have."""
        return scalar_to_dtype(scalar, self._class_ids.dtype)

    def parse_class_words_file(self, zip_file=None):
        """Parse the file containing the class descriptions in the
        internal zip file or `zip_file` if given.
        """
        self._require_zip_file_set(zip_file)

        # Find word for ID
        # Because the words.txt list contains _all_ ImageNet classes, we need
        # to filter manually.
        with self._zip_file.open(
                'tiny-imagenet-200/words.txt'
        ) as class_words_file:
            self.parse_class_words(class_words_file)

    def parse_class_words(self, class_words_file):
        """Parse all class descriptions in `class_words_file`."""
        lines = map(rstrip_newline, class_words_file)
        # Class IDs come sorted from lowest to highest
        cmp_class_id_index = 0

        for line in lines:
            (class_id, words) = line.split(b'\t')
            class_id = self.to_class_id_dtype(class_id)

            # Skip until we have the correct class ID
            cmp_class_id = self._class_ids[cmp_class_id_index]
            while (cmp_class_id_index < len(self._class_ids) - 1
                   and cmp_class_id < class_id):
                cmp_class_id_index += 1
                cmp_class_id = self._class_ids[cmp_class_id_index]

            # The dataset doesn't seem to use this class
            if cmp_class_id != class_id:
                continue
            self._class_words[cmp_class_id_index] = words
