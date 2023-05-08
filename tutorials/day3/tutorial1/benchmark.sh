#!/bin/bash

if [[ $# -ne 2 ]] ; then
    echo 'Usage: ./benchmark.sh config.csv output.csv'
    echo '- config.csv contains the configurations to try (check config_example.csv as an example)'
    echo '- output.csv contains the output filename where results will be stored'
    exit 0
fi
source /p/project/training2306/software_environment/activate.sh
ACCOUNT=training2306
TIME=00:10:00
PARTITION=booster
CPUS_PER_TASK=24
export OMP_NUM_THREADS=24
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export HOROVOD_AUTOTUNE=1
export HOROVOD_HIERARCHICAL_ALLGATHER=0
export HOROVOD_HIERARCHICAL_ALLREDUCE=0
export TF_CUDNN_USE_AUTOTUNE=1
export TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT=1
export TF_AUTOTUNE_THRESHOLD=2

CONFIG=$1
OUTPUT=$2
# Add missing newline at the end of the file if it is missing (otherwise, keep the contents as is)
# This is necessary to ignore the header when parsing the CSV
config_contents="$(sed -e '$a\' $CONFIG)"

echo "NODES,LOCAL_BATCH_SIZE,GPUS_PER_NODE,TOTAL_IMAGES_PER_SEC">$OUTPUT

# try different configuration of number of nodes and batch size using $CONFIG
# and write the result to $OUTPUT
nb_lines=$(echo "$config_contents"|wc -l)
nb_lines=$((nb_lines-1)) # to ignore header
for line in $(echo "$config_contents"|tail -n $nb_lines);do
    NODES=$(echo $line|cut -d, -f1)
    GPUS_PER_NODE=$(echo $line|cut -d, -f2)
    LOCAL_BATCH_SIZE=$(echo $line|cut -d, -f3)
    RUN_CONFIG="NODES_${NODES}_LOCAL_BATCH_SIZE_${LOCAL_BATCH_SIZE}_GPUS_PER_NODE_${GPUS_PER_NODE}"
    echo "Running configuration: $RUN_CONFIG"
    # for each configuration, the standard output will be written in $STDOUT
    STDOUT="${RUN_CONFIG}.out"
    # do the run for the current configuration
    srun --output=$STDOUT --error=$STDOUT -N $NODES -n $((GPUS_PER_NODE*NODES)) --ntasks-per-node=$GPUS_PER_NODE --account=$ACCOUNT --cpus-per-task=$CPUS_PER_TASK --time=$TIME --gres=gpu:$GPUS_PER_NODE --partition=$PARTITION --cpu-bind=none,v --accel-bind=gn python -u train.py --batch_size=$LOCAL_BATCH_SIZE

    # extract the throughput from the standard output file and add it to `results.csv`
    total_images_per_sec=$(cat $STDOUT|grep "total images/sec:"|cut -d ":" -f 2)
    echo "Total images per sec: $total_images_per_sec"
    echo "$NODES,$LOCAL_BATCH_SIZE,$GPUS_PER_NODE,$total_images_per_sec">>$OUTPUT
done
echo "Total time spent: $SECONDS secs"
