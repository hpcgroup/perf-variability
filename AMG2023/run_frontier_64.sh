#!/bin/bash
#SBATCH -N 64
#SBATCH -n 512
#SBATCH -q normal
#SBATCH -J amg
#SBATCH --gpu-bind none
#SBATCH -t 00:30:00
#SBATCH -A csc569
#SBATCH --output /lustre/orion/csc569/scratch/keshprad/perfvar/AMG2023_logs/64nodes/%x-%j/output-AMG2023.log
#SBATCH --error /lustre/orion/csc569/scratch/keshprad/perfvar/AMG2023_logs/64nodes/%x-%j/error-AMG2023.log
#SBATCH --exclusive
# Run like: sbatch run_frontier_64.sh

OUTPUT_DIR=/lustre/orion/csc569/scratch/keshprad/perfvar/AMG2023_logs/64nodes/$SLURM_JOB_NAME-$SLURM_JOB_ID
OUTPUT_FILE=$OUTPUT_DIR/output-AMG2023.log
ERROR_FILE=$OUTPUT_DIR/error-AMG2023.log

# Run gpu benchmarks
COMM_TYPE=mpi
PERF_VARIABILITY_ROOT=/ccs/home/keshprad/perf-variability
echo running allreduce benchmark
bash $PERF_VARIABILITY_ROOT/gpu-benchmarks/allreduce/run_frontier.sh $COMM_TYPE $SLURM_JOB_NUM_NODES $OUTPUT_DIR
# echo running allgather benchmark
# bash $PERF_VARIABILITY_ROOT/gpu-benchmarks/allgather/run_frontier.sh $COMM_TYPE $SLURM_JOB_NUM_NODES $OUTPUT_DIR
echo running gemm benchmark
bash $PERF_VARIABILITY_ROOT/gpu-benchmarks/gemm/run_frontier.sh $SLURM_JOB_NUM_NODES $OUTPUT_DIR

APP_ROOT=/ccs/home/keshprad/AMG2023
cd $APP_ROOT

# reset modules
echo resetting modules:
module reset
# load modules
echo loading modules:
module load cray-mpich/8.1.30
module load craype-accel-amd-gfx90a
module load rocm/6.1.3

export MPICH_GPU_SUPPORT_ENABLED=1
export CRAY_ACCEL_TARGET=gfx90a
export HYPRE_INSTALL_DIR=/ccs/home/keshprad/hypre/src/hypre/
# mpiP
export LD_LIBRARY_PATH=/ccs/home/keshprad/mpiP:$LD_LIBRARY_PATH
export MPIP="-o -f $OUTPUT_DIR"

# log start date
echo start AMG2023: $(date)
# define command
cmd="srun --output $OUTPUT_FILE --error $ERROR_FILE \
        ./build/amg -P 8 8 8 -n 128 64 64 -problem 1 -iter 500"
echo solving:
echo $cmd
$cmd
# log end date
echo end AMG2023: $(date)
