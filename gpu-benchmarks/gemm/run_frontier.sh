# This script assumes it is being run by another sbatch script, 
# so does not include portions for SBATCH vars (e.g. account, time, etc.)

# run like: bash /ccs/home/keshprad/gpu-benchmarks/benchmark/frontier/gemm.sh <num_nodes> <output_dir>

#!/bin/bash
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <number_of_nodes> <output_dir>"
    exit 1
fi
# `16` or `64`
NUM_NODES=$1
# output directory
OUTPUT_DIR=$2

OUTPUT_FILE=$OUTPUT_DIR/output-gemm.log

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
    EXEC=$GPU_BENCHMARKS_ROOT/matmul/frontier/gemm.x
    NUM_TASKS=$(($NUM_NODES * 8))

    export MPICH_GPU_SUPPORT_ENABLED=1
    export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}"

    echo start gemm: $(date)
    CMD="srun -N $NUM_NODES -n $NUM_TASKS \
            --gpus-per-node 8 \
            --gpus-per-task 1 \
            --ntasks-per-node 8 \
            --output $OUTPUT_FILE \
            $EXEC"
    echo running:
    echo $CMD
    $CMD
    echo end gemm: $(date)
} >> $OUTPUT_FILE
