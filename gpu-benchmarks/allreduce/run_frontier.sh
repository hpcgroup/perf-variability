# This script assumes it is being run by another sbatch script, 
# so does not include portions for SBATCH vars (e.g. account, time, etc.)

# run like: bash /ccs/home/keshprad/gpu-benchmarks/benchmark/frontier/allreduce.sh <comm_type> <num_nodes> <output_dir>

#!/bin/bash
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <communication_type> <number_of_nodes> <output_dir>"
    exit 1
fi
# `mpi` or `rccl`
COMM_TYPE=$1
# `16` or `64`
NUM_NODES=$2
# output directory
OUTPUT_DIR=$3

OUTPUT_FILE=$OUTPUT_DIR/output-allreduce.log

{
    # reset modules
    echo resetting modules:
    module reset
    # load modules
    echo loading modules:
    module load PrgEnv-cray craype-accel-amd-gfx90a cpe/23.05 amd/6.1.3
    module load cray-mpich/8.1.30
    module load rocm/6.1.3

    GPU_BENCHMARKS_ROOT=/lustre/orion/csc569/scratch/keshprad/gpu-benchmarks
    EXEC=$GPU_BENCHMARKS_ROOT/allreduce_$COMM_TYPE.x
    NUM_TASKS=$(($NUM_NODES * 8))
    MIN_MSG_SIZE=$((1 * 1024))
    MAX_MSG_SIZE=$((1 * 1024 * 1024))
    ITERATIONS=100

    export MPICH_GPU_SUPPORT_ENABLED=1
    export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}"

    echo start allreduce: $(date)
    CMD="srun -N $NUM_NODES -n $NUM_TASKS \
            --output $OUTPUT_FILE \
            $EXEC $NUM_TASKS $MIN_MSG_SIZE $MAX_MSG_SIZE $ITERATIONS"
    echo running:
    echo $CMD
    $CMD
    echo end allreduce: $(date)
} >> $OUTPUT_FILE
