from mpi4py import MPI

comm = MPI.COMM_WORLD

# Solution to task 1
rank = comm.Get_rank()
print(f'I am a unique process with rank {rank}.')

# Solution to task 2
ROOT_RANK = 0
if rank == ROOT_RANK:
    size = comm.Get_size()
    print(f'Process {rank} here, we are {size} processes in total!')
