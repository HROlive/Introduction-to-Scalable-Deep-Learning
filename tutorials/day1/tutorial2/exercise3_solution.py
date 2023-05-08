"""
Calculate a histogram in parallel over a computer vision dataset.

Even though we iterate over a primarily RGB dataset, we handle
greyscale images by treating them like an RGB image (meaning we assign
the same intensity for all channels).

This version is kept simpler but misses functionality to calculate
histograms for each individual class.
"""

import math
from pathlib import Path
import time
import zipfile

from mpi4py import MPI
import numpy as np
from PIL import Image


# `MPI.Init` automatically called upon import

DATASET_PATH = Path('/p/project/training2306/datasets/tiny-imagenet-200.zip')

NUM_INTENSITIES = 256
NUM_CHANNELS = 3
EXPECTED_MODES = ['RGB', 'L']

ROOT_RANK = 0


def update_histograms(histograms, image_file, image_path):
    """
    Update the `histograms` matrix using the data in `image_file`.
    `image_path` contains the path to the image.
    """
    # We require the path explicitly because Pillow cannot grab a path from a
    # non-existent file.
    with Image.open(image_file) as image:
        # Skip images with unexpected mode
        if image.mode not in EXPECTED_MODES:
            print('Unexpected mode for {}: {}'.format(image_path, image.mode))
            return

        # The histograms for the channels red, green and blue
        # are concatenated.
        concatenated_histograms = np.array(
            image.histogram(),
            dtype=np.int,
        )

        if image.layers == 1:
            # We handle greyscale images by treating them like an RGB
            # image (meaning we assign the same intensity for
            # all channels).
            shaped_histograms = concatenated_histograms
        else:
            shaped_histograms = concatenated_histograms.reshape(
                NUM_CHANNELS,
                NUM_INTENSITIES,
            )

        histograms += shaped_histograms


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == ROOT_RANK:
        start_time = time.perf_counter()

    # These will be in RGB order, so one histogram for the red channel, one for
    # the blue channel and one for the green channel.
    histograms = np.zeros((NUM_CHANNELS, NUM_INTENSITIES), dtype=np.int)

    with zipfile.ZipFile(DATASET_PATH, 'r') as zipped_data:
        image_names = list(filter(
            lambda name: name.endswith('.JPEG'),
            zipped_data.namelist(),
        ))

        # Task 1 solution
        num_divided_images = math.ceil(len(image_names) / size)
        offset = rank * num_divided_images
        image_names_shard = image_names[offset:offset + num_divided_images]

        for image_name in image_names_shard:
            with zipped_data.open(image_name) as image_file:
                update_histograms(histograms, image_file, image_name)

    collected_histograms = np.zeros(
        (NUM_CHANNELS, NUM_INTENSITIES),
        dtype=np.int,
    )

    # Task 2 solution
    comm.Reduce(
        [histograms, MPI.INT],
        [collected_histograms, MPI.INT],
        op=MPI.SUM,
        root=ROOT_RANK,
    )

    if rank == ROOT_RANK:
        end_time = time.perf_counter()

        print(collected_histograms)
        print(end_time - start_time)
        np.save(Path('histograms.npy'), collected_histograms)
        with open(Path('runtime.out'), 'w') as runtime_file:
            runtime_file.write('{}\n'.format(end_time - start_time))


if __name__ == '__main__':
    main()

# `MPI.Finalize` handler automatically registered upon import
