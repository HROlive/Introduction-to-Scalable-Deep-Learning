"""
Wrapper for TensorFlow datasets to track the pipeline.

Each method call is passed to the stored dataset and tracked if
successful.
"""

import inspect

from tensorflow.data import Dataset


class TFDataset(Dataset):
    """Wrapper for TensorFlow datasets to track the pipeline.

    Each standard method call is passed to the stored dataset and
    tracked if successful. The pipeline can be accessed with the
    `pipeline` property; the dataset with the `dataset` property.

    Each `iter` call is also tracked and can be accessed as
    `num_iterations`.
    """

    def __init__(self, dataset):
        if not isinstance(dataset, Dataset):
            raise TypeError('Cannot wrap a non-TensorFlow dataset.')

        self.__dataset = dataset
        self.__pipeline = []
        self.__wrap_dataset()

        self.__num_iterations = 0

    def __bool__(self):
        return bool(self.__dataset)

    def __len__(self):
        """Return the length of the wrapped dataset."""
        return len(self.__dataset)

    def __iter__(self):
        """Iterate the wrapped dataset."""
        iter_ = iter(self.__dataset)
        self.__num_iterations += 1
        return iter_

    # We need to provide this to implement the abstract `tf.data.Dataset`.
    def _inputs(self):
        return self.__dataset._inputs()

    # We need to provide this to implement the abstract `tf.data.Dataset`.
    @property
    def element_spec(self):
        return self.__dataset.element_spec

    # Static methods from `tf.data.Dataset`

    @staticmethod
    def from_generator(*args, **kwargs):
        dataset = Dataset.from_generator(*args, **kwargs)

        new_self = TFDataset(dataset)
        call_str = 'from_generator' + TFDataset.__format_args(args, kwargs)
        new_self.__pipeline.append(call_str)
        return new_self

    @staticmethod
    def from_tensor_slices(*args, **kwargs):
        dataset = Dataset.from_tensor_slices(*args, **kwargs)

        new_self = TFDataset(dataset)
        call_str = 'from_tensor_slices([...])'
        new_self.__pipeline.append(call_str)
        return new_self

    @staticmethod
    def from_tensors(*args, **kwargs):
        dataset = Dataset.from_tensors(*args, **kwargs)

        new_self = TFDataset(dataset)
        call_str = 'from_tensors([...])'
        new_self.__pipeline.append(call_str)
        return new_self

    @staticmethod
    def list_files(*args, **kwargs):
        dataset = Dataset.list_files(*args, **kwargs)

        new_self = TFDataset(dataset)
        call_str = 'list_files([...])'
        new_self.__pipeline.append(call_str)
        return new_self

    @staticmethod
    def range(*args, **kwargs):
        dataset = Dataset.range(*args, **kwargs)

        new_self = TFDataset(dataset)
        call_str = 'range' + TFDataset.__format_args(args, kwargs)
        new_self.__pipeline.append(call_str)
        return new_self

    @staticmethod
    def zip(datasets, *args, **kwargs):
        dataset = Dataset.zip(datasets, *args, **kwargs)

        new_self = TFDataset(dataset)
        # We try to give the 'zip' effect by creating a tuple of the
        # zipped pipelines
        new_self.__pipeline.append(tuple(map(lambda d: d.pipeline, datasets)))
        call_str = f'zip(<{len(datasets)} datasets>)'
        new_self.__pipeline.append(call_str)
        return new_self

    # Wrapping code

    @staticmethod
    def __preprocess_arg(arg):
        if inspect.isfunction(arg):
            if arg.__module__ == '__main__':
                return arg.__name__
            return arg.__module__ + '.' + arg.__name__
        return repr(arg)

    @staticmethod
    def __argformat(args):
        """Return `tuple` `args` formatted as if passed to a function.

        This mostly entails that tuples of length 1 will not have a
        trailing comma.
        """
        if len(args) != 1:
            return str(TFDataset.__preprocess_arg(args))

        preprocessed = tuple(map(TFDataset.__preprocess_arg, args))
        args_str = ', '.join(preprocessed)
        return '(' + args_str + ')'

    @staticmethod
    def __kwargformat(kwargs):
        """Return `dict` `kwargs` formatted as if passed to a function."""
        kwargs = [f'{key}={TFDataset.__preprocess_arg(value)}'
                  for (key, value) in kwargs.items()]
        return ', '.join(kwargs)

    @staticmethod
    def __format_args(args, kwargs):
        """Format the given `tuple` `args` and `dict` `kwargs` so they look as
        if a function was called with them.
        Includes the parenthesis around the arguments.
        """
        args_str = TFDataset.__argformat(args)
        # Remove trailing parenthesis
        args_str = args_str[:-1]
        kwargs_str = TFDataset.__kwargformat(kwargs)
        separator = ', ' if args and kwargs else ''
        return f'{args_str}{separator}{kwargs_str})'

    def __wrapped_function(self, function, name):
        """Return `function` wrapped so we track calls to it by its `name`."""
        def wrapper(*args, **kwargs):
            dataset = function(*args, **kwargs)

            # Handle functions which do not return a dataset
            if not isinstance(dataset, Dataset):
                return dataset

            new_self = TFDataset(dataset)
            new_self.__pipeline = self.__pipeline.copy()
            call_str = name + self.__format_args(args, kwargs)
            new_self.__pipeline.append(call_str)
            return new_self

        wrapper.__name__ = name
        if function.__doc__:
            orig_signature = str(inspect.signature(function))
            wrapper.__doc__ = (f'Original signature:\n'
                               f'{name}{orig_signature}\n\n'
                               f'{function.__doc__}')
        return wrapper

    def __wrap_dataset(self):
        """Wrap all methods of the stored dataset in `self`.

        Note that this does _not_ wrap dunder methods, so these should
        be written or wrapped manually as required.
        """
        for (name, method) in inspect.getmembers(
                self.__dataset,
                inspect.ismethod,
        ):
            if name.startswith('__') and name.endswith('__'):
                continue
            setattr(self, name, self.__wrapped_function(method, name))

    def __getattr__(self, name):
        return getattr(self.__dataset, name)

    @property
    def dataset(self):
        """Return the stored dataset."""
        return self.__dataset

    @property
    def pipeline(self):
        """Return the pipeline traced until now."""
        return self.__pipeline

    @property
    def num_iterations(self):
        """Return the amount of times `iter` was called on `self`."""
        return self.__num_iterations
