from pathlib import Path
import time

import matplotlib
matplotlib.use('pdf')  # noqa
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import tiny_imagenet_loader
# Task: Add import statement for `horovod.tensorflow.keras` here.
#       Remember, canonically it is imported as `hvd`.
import horovod.tensorflow.keras as hvd

# Task: Add init statement for Horovod here.
hvd.init()

# Set up GPU
gpus = tf.config.list_physical_devices('GPU')
# Currently, memory growth needs to be the same across GPUs.
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    # Task: Use Horovod local rank to pick GPU.
    tf.config.set_visible_devices(gpus[hvd.local_rank()], 'GPU')


def load_dataset(
        path,
        batch_size,
        num_classes,
        num_taken=None,
        is_train=False,
        seed=None,
):
    """Load the dataset at `path` with the given `batch_size`.

    `num_classes` indicates the number of classes in the dataset.
    `num_taken` is the amount of data (rows) to `take` from the
    dataset.
    `is_train` specifies whether we are handling the train
    dataset.
    `seed` gives the random seed for shuffling. Only relevant
    if `is_train is True`.
    """
    dataset = tiny_imagenet_loader.from_tfrecords(path)
    if num_taken is not None:
        dataset = dataset.take(num_taken)

    # Task: Shard the dataset.
    #       `tf.data` datasets have a `shard` method that takes the
    #       total amount of workers as the first argument so it knows
    #       how large each shard is. As the second argument, it requires
    #       some kind of index to know which shard to assign to this
    #       worker.
    #       Hint: It's not the local rank!
    dataset = dataset.shard(hvd.size(), hvd.rank())

    # Bonus task: Normalize the images so their values are between 0 and 1
    #             inclusively.
    #             The colors have a maximum intensity of 255.
    dataset = dataset.map(
        lambda image, label: (image / 255, label)
    )

    # One-hot encode our labels due to the model we use
    dataset = dataset.map(
        lambda image, label: (image, tf.one_hot(label, num_classes))
    )

    if is_train:
        dataset = dataset.shuffle(5000, seed=seed)

    dataset = dataset.batch(batch_size)
    return dataset


# Training hyperparameters: local batch size and (global) learning rate
batch_size = 64
learning_rate = 0.01
# Random seed; for sanity during experimentation/debugging, you may
# want to keep this fixed.
seed = 0

# Construct datasets
num_classes = 201  # 200 classes in the dataset plus an unknown class.

# We train only on subsets of the datasets, to keep the runtime short.
# You can set `take_data_ratio = 1` if you want to train on the whole
# datasets and accept waiting longer.
take_data_ratio = 0.2
num_train_data = int(100000 * take_data_ratio)
num_val_data = int(10000 * take_data_ratio)
num_test_data = int(10000 * take_data_ratio)

train = load_dataset(
    Path('/p/project/training2306/datasets/tiny-imagenet-200-train.tfrecords'),
    batch_size,
    num_classes,
    num_taken=num_train_data,
    is_train=True,
    seed=seed,
)
val = load_dataset(
    Path('/p/project/training2306/datasets/tiny-imagenet-200-val.tfrecords'),
    batch_size,
    num_classes,
    num_taken=num_val_data,
)
test = load_dataset(
    Path('/p/project/training2306/datasets/tiny-imagenet-200-test.tfrecords'),
    batch_size,
    num_classes,
    num_taken=num_test_data,
)


# Construct model and optimizer
model = tf.keras.applications.resnet50.ResNet50(
    weights=None,
    classes=num_classes
)

# Defining the optimizer
optimizer = tf.keras.optimizers.SGD(
    learning_rate=learning_rate,
    momentum=0.9,
    nesterov=True,
)

# Task: Add the Horovod wrapper of the optimizer here.
optimizer = hvd.DistributedOptimizer(optimizer)

model.compile(
    loss=tf.keras.losses.categorical_crossentropy,
    optimizer=optimizer,
    metrics=[tf.keras.metrics.CategoricalAccuracy()],
)


# Training
callbacks = [
    # Task: Uncomment the callbacks to make sure global variables are
    #       broadcasted in the beginning and metrics are averaged at
    #       the end of each epoch.
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),
    hvd.callbacks.MetricAverageCallback(),
]
start = time.time()
epochs = 5

# Try to find out if we are the root node.
# If `hvd` is not available, it will always set `root_process` to `True`.
try:
    if hvd.rank() == 0:
        root_process = True
    else:
        root_process = False
    hvd_size = hvd.size()
except NameError:  # Horovd is not available as `hvd`
    root_process = True
    hvd_size = 1

history = model.fit(
    train,
    epochs=epochs,
    validation_data=val,
    callbacks=callbacks,
    verbose=2 if root_process else 0,
)
duration = time.time() - start


# Display results
def plot_accuracy(history, plot_filename):
    """Plot training accuracy from the given `history` object.
    Store the plot at `plot_filename`.
    """
    (fig, (ax0, ax1)) = plt.subplots(nrows=2, sharex=True, figsize=(12, 6))

    history_df = pd.DataFrame(history.history)
    history_df['epoch'] = history.epoch
    history_df.plot(x='epoch', y=['loss', 'val_loss'], ax=ax0)
    history_df.plot(
        x='epoch',
        y=['categorical_accuracy', 'val_categorical_accuracy'],
        ax=ax1,
    )
    fig.savefig(plot_filename)


if root_process:
    plot_accuracy(history, 'loss_accuracy_plot.pdf')

    num_train_images_processed = num_train_data * epochs
    print(f'total training images: {num_train_images_processed}')
    print(f'total training time in sec: {duration}')
    print(f'total images per second during training: '
          f'{num_train_images_processed / duration}')

    print('\nTesting our model on unseen data...')


# Model evaluation
(loss, accuracy) = model.evaluate(
    test,
    callbacks=callbacks,
    verbose=2 if root_process else 0,
)

if root_process:
    print(f'Our model had a test loss of {loss:.4f}.')
    print(f'Its test accuracy was {accuracy:.4f}.')
