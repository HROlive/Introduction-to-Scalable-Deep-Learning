#!/usr/bin/env bash

# This `sbatch` script provides a template for using a FUSE-mounted
# SquashFS.
#
# Places that need adaptation depending on your needs are marked with
# "TODO".
#
# The advantage of using a SquashFS is good multiprocessing
# performance with arbitrarily sized data.
# The disadvantage is that FUSE is noticably slower than reading
# directly from RAM. If your data fits into RAM, prefer `shm.sbatch`
# instead.
# Another disadvantage is that a SquashFS is read-only. If your data
# changes, you will have to re-create the SquashFS every time.
# Similarly, you cannot write to the same location where you read data
# from.

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=# TODO
#SBATCH --account=training2306
#SBATCH --signal=TERM@60  # This allows us to clean up the FUSE.
#SBATCH --partition=dc-gpu

[...]  # TODO

# Path to your data.
export DATA_PATH="..."  # TODO
# Where to create the SquashFS.
export SQSH_PATH="$DATA_PATH.sqsh"
# Where to put the virtual file system that accesses the SquashFS.
# Has to be in `/dev/shm` or `/tmp` on Jülich machines.
export MOUNT_PATH="/dev/shm/$(whoami)/sqsh/$(basename "$DATA_PATH")"

# Create the SquashFS if it doesn't exist yet.
[ -e "$SQSH_PATH" ] || mksquashfs "$DATA_PATH" "$SQSH_PATH"
# Optionally, to use a larger block size (may increase performance when
# you have larger files), replace the above line with:
# [ -e "$SQSH_PATH" ] || mksquashfs "$DATA_PATH" "$SQSH_PATH" -b 1M

unmount_squashfuse() {
    # Do nothing on tasks with node-local rank other than 0.
    ((SLURM_LOCALID)) && return 0
    [ -d "$MOUNT_PATH" ] && fusermount3 -u "$MOUNT_PATH"
    rm -rf "$MOUNT_PATH"
}
export -f unmount_squashfuse

mount_squashfuse() {
    # Do nothing on tasks with node-local rank other than 0.
    ((SLURM_LOCALID)) && return 0

    [ -d "$MOUNT_PATH" ] && ls -l "$MOUNT_PATH"
    # We unmount and completely remove `MOUNT_PATH` to prevent dirty
    # data (maybe there are still some remains from previous jobs).
    [ -d "$MOUNT_PATH" ] && fusermount3 -u "$MOUNT_PATH" || true
    rm -rf "$MOUNT_PATH"
    # Create a private `MOUNT_PATH`.
    mkdir -m 700 -p "$MOUNT_PATH"

    # Register a clean-up handler for `MOUNT_PATH` that will run once this
    # job finishes, is interrupted, times out, or receives `scancel`.
    trap 'bash -c unmount_squashfuse' EXIT SIGINT SIGTERM SIGCONT
    # Try to make the SquashFS available at `MOUNT_PATH`. If that
    # fails, exit.
    squashfuse_ll "$SQSH_PATH" "$MOUNT_PATH" || exit 1

    # Enter an infinite loop so we can still receive signals in this
    # process. (The `srun` that calls this doesn't exit anyway.)
    while true; do
        # 25 hours
        sleep 90000
    done
}
export -f mount_squashfuse

wait_for_mount() {
    # Get the process ID of the most recent mount process.
    mount_pid="$(pgrep -n -f -u "$(whoami)" -- ' -c mount_squashfuse$')"
    # Check whether the mount process died or whether the mount is
    # complete.
    while ps -p "$mount_pid" > /dev/null \
            && ! mountpoint -q "$MOUNT_PATH"; do
        sleep 1
    done
}
export -f wait_for_mount

# Mount the SquashFS at `MOUNT_PATH`.
#
# This `srun` will never stop running (barring errors) due to the FUSE
# mount. We put it in the background and allow other jobs to overlap
# with its resources.
srun --overlap bash -c mount_squashfuse &

# Make sure the mount is complete on all nodes before continuing. This
# is required because we had to put the previous `srun` in the
# background.
srun bash -c wait_for_mount

# Access the SquashFS at `MOUNT_PATH`.
srun python main.py --input_path="$MOUNT_PATH"  # TODO
