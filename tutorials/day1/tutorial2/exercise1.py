from mpi4py import MPI

comm = MPI.COMM_WORLD

# Task 1: Please extend this so each process gets the attention it
#         deserves: Have each process print its rank. Use the
#         communicator's `Get_rank()` method.

# Task 2: Let a single "root" process summarize how many processes
#         exist in total. Use the communicator's `Get_size()` method.
