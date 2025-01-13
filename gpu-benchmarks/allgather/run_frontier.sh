# This script assumes it is being run by another sbatch script, 
# so does not include portions for SBATCH vars (e.g. account, time, etc.)

# run like: bash /ccs/home/keshprad/gpu-benchmarks/benchmark/frontier/allgather.sh <comm_type> <num_nodes> <output_dir>

#!/bin/bash
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <communication_type> <rocm_version> <number_of_nodes> <output_dir>"
    exit 1
fi
# `mpi` or `rccl`
COMM_TYPE=$1
# `5.7.1` or `6.1.3`
ROCM_VERSION=$2
# `16` or `64`
NUM_NODES=$3
# output directory
OUTPUT_DIR=$4

# setup cray-mpich version
if [[ "$ROCM_VERSION" == "6.1.3" ]]; then
    MPICH_VERSION=8.1.30
else
    MPICH_VERSION=8.1.28
fi

OUTPUT_FILE=$OUTPUT_DIR/output-allgather.log

{
    # reset modules
    echo resetting modules:
    module reset
    # load modules
    echo loading modules:
    module load PrgEnv-cray craype-accel-amd-gfx90a cpe/23.05 amd/${ROCM_VERSION}
    module load cray-mpich/${MPICH_VERSION}
    module load rocm/${ROCM_VERSION}
    module list

    GPU_BENCHMARKS_ROOT=/lustre/orion/csc569/scratch/keshprad/gpu-benchmarks
    EXEC=$GPU_BENCHMARKS_ROOT/allgather_$COMM_TYPE\_rocm-${ROCM_VERSION}.x
    NUM_TASKS=$(($NUM_NODES * 8))
    MIN_MSG_SIZE=$((1 * 1024))
    MAX_MSG_SIZE=$((1 * 1024 * 1024))
    ITERATIONS=100

    export MPICH_GPU_SUPPORT_ENABLED=1
    export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}"

    echo start allgather: $(date)
    For MPI-bench we should use --gpus-per-node --gpus-per-task --ntasks-per-node , and  --gpu-bind=none in srun.
    CMD="srun -N $NUM_NODES -n $NUM_TASKS \
            --gpus-per-node 8 \
            --gpus-per-task 1 \
            --ntasks-per-node 8 \
            --gpu-bind none \
            --output $OUTPUT_FILE \
            $EXEC $NUM_TASKS $MIN_MSG_SIZE $MAX_MSG_SIZE $ITERATIONS"
    echo running:
    echo $CMD
    $CMD
    echo end allgather: $(date)
} &>> $OUTPUT_FILE
