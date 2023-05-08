import math
import sys
import time

import matplotlib
matplotlib.use('pdf')  # noqa
import matplotlib.pyplot as plt
from mpi4py import MPI
import numpy as np

ROOT_RANK = 0


def plot_results(size, total_num_samples, num_samples, all_samples):
    # Do not plot results if we cannot see anything due to the
    # amount of samples.
    if total_num_samples > 1000000:
        return

    plt.figure(figsize=(8, 8))
    plt.xlim((0, 1))
    plt.ylim((0, 1))

    unit_circle = plt.Circle((0, 0), 1, alpha=0.1)
    plt.gca().add_artist(unit_circle)

    # If we only have a few samples, seeing them as pixels becomes
    # harder, so use a larger marker then.
    marker = '.' if total_num_samples < 2000 else ','

    for i in range(size):
        proc_i = i * num_samples
        proc_samples = all_samples[proc_i:proc_i + num_samples, :]

        # We plot the samples from each process individually so
        # they get a distinct color.
        plt.plot(proc_samples[:, 0], proc_samples[:, 1], marker)

    plt.savefig('plot.pdf')


def main():
    total_num_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 100000

    start_time = time.perf_counter()

    comm = MPI.COMM_WORLD

    rank = comm.Get_rank()
    size = comm.Get_size()

    # For simplicity, we do not care about getting exactly
    # `total_num_samples`. If `total_num_samples` is not divisible by
    # `size`, we get a bit more.
    num_samples = math.ceil(total_num_samples / size)
    if rank == ROOT_RANK and num_samples != total_num_samples / size:
        print(f'Warning: We will have {num_samples * size} instead of '
              f'{total_num_samples} samples.\n')
        total_num_samples = num_samples * size

    rng = np.random.default_rng()
    samples = rng.random((num_samples, 2))
    squared_samples = np.power(samples, 2)
    num_in_circle = np.count_nonzero(
        # Standard unit circle test: x^2 + y^2 <= 1
        squared_samples[:, 0] + squared_samples[:, 1] <= 1
    )
    # Wrap in NumPy array so MPI can handle it automatically.
    num_in_circle = np.array(num_in_circle)

    if rank == ROOT_RANK:
        # Set up receive buffers
        total_num_in_circle = np.array(0)
        all_samples = np.empty((total_num_samples, 2), dtype=samples.dtype)
    else:
        total_num_in_circle = None
        all_samples = None

    # Solution for task 1 in `exercise2_solution.sbatch`

    # Solution for task 2
    comm.Reduce(
        num_in_circle,
        total_num_in_circle,
        op=MPI.SUM,
        root=ROOT_RANK,
    )

    # Solution for task 3
    comm.Gather(
        samples,
        all_samples,
        root=ROOT_RANK,
    )

    if rank == ROOT_RANK or total_num_samples == 0:
        ratio = total_num_in_circle / total_num_samples

        calc_duration = time.perf_counter() - start_time
        print(f'Results obtained after {calc_duration:.5f} seconds.\n'
              f'Total number of samples:          {total_num_samples}\n'
              f'Number of samples in unit circle: '
              f'{total_num_in_circle:>{len(str(total_num_samples))}}\n'
              f'Circle-to-square ratio: {ratio}\n'
              f'Ratio times four:       {ratio * 4}\n'
              f'Difference to π:        {abs(ratio * 4 - math.pi):.5f}')

        # Handle task 3 not being solved
        try:
            all_samples
        except NameError:
            return

        plot_results(size, total_num_samples, num_samples, all_samples)


if __name__ == '__main__':
    main()
