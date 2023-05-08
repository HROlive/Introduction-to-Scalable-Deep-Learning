import tensorflow as tf
# The import statement for other Horovod-supported libraries like
# PyTorch or MXNet is very similar.
import horovod.tensorflow.keras as hvd


# GPU Initialization

# We need to inform each worker which GPU it will get. Here we make
# use of Horovod's `local_rank` method.

def initialize():
    """Initialize Horovod and the GPUs."""
    # Any code using Horovod needs to initialize it.
    hvd.init()

    # Get a list of all physically available GPUs on the node.
    gpus = tf.config.list_physical_devices('GPU')

    # Use worker's local rank to determine which GPU it will use.
    # Every worker will thus get its own GPU for exclusive use.
    gpu_id = hvd.local_rank()

    # Take care that TensorFlow does not occupy all GPU memory.
    # Currently, memory growth needs to be the same for all devices.
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        # Task: Pin a GPU to the worker using the worker's local rank.
        #       Hint: You need to pass a GPU object here, not just
        #       an index.
        tf.config.set_visible_devices(# TODO, 'GPU')


# We print basic info about the global environment and the worker's
# basic rank variables, which will make the basic structure of a
# Horovod distributed training run clear.
#
# Use functions like `hvd.rank()`, `hvd.size()`, or `hvd.local_rank()` here.
def main():

    # Task: Determine the worker's basic rank variables.
    rank = # TODO
    local_rank = # TODO

    # Task: Determine the total and local number of workers.
    world_size = # TODO
    local_size = # TODO

    # Task: Print generic non-worker-specific info (e.g. TensorFlow
    #       version, world size, ...) using only the worker with
    #       rank 0:
    # TODO

    # Task: Print worker-specific info: its local rank and world size.
    # TODO


if __name__ == '__main__':
    initialize()
    main()
