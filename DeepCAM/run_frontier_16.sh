#!/bin/bash
#SBATCH -N 16
#SBATCH -n 128
#SBATCH -q normal
#SBATCH -J deepcam
#SBATCH --gpu-bind none
#SBATCH -t 00:30:00
#SBATCH -A csc569
#SBATCH --output /lustre/orion/csc569/scratch/keshprad/perfvar/deepcam_logs/16nodes/%x-%j/job-output.log
#SBATCH --error /lustre/orion/csc569/scratch/keshprad/perfvar/deepcam_logs/16nodes/%x-%j/job-error.log
#SBATCH --exclusive
# Run like: sbatch run_frontier_16.sh

echo "start run: $(date)"
export JOB_OUTPUT_PATH=/lustre/orion/csc569/scratch/keshprad/perfvar/deepcam_logs/16nodes/$SLURM_JOB_NAME-$SLURM_JOB_ID
OUTPUT_FILE=${JOB_OUTPUT_PATH}/output-deepcam.log
ERROR_FILE=${JOB_OUTPUT_PATH}/error-deepcam.log

export SCRATCH="/lustre/orion/csc569/scratch/keshprad"
export APP_ROOT="${SCRATCH}/deepcam"
APP_WORKING_DIR=${APP_ROOT}/hpc/deepcam/src/deepCam
cd $APP_WORKING_DIR

# load modules
ROCM_VERSION=6.1.3
echo resetting modules:
module reset
echo loading modules:
module load PrgEnv-gnu/8.5.0
module load rocm/6.1.3
module load craype-accel-amd-gfx90a
module load cray-python/3.9.13.1
module list

# activate virtual env
echo activating virtual env:
source ${APP_ROOT}/.venv/bin/activate

# ENV variables
echo setting env vars:
NNODES=$SLURM_JOB_NUM_NODES
GPUS=$(( NNODES * 8 ))

## master addr and port
# setting variables for torch.distributed
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export WORLD_SIZE=$GPUS
export OMP_NUM_THREADS=7 

# Needed to bypass MIOpen, Disk I/O Errors
export MIOPEN_USER_DB_PATH="/tmp/my-miopen-cache-${SLURM_JOB_ID}"
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}

## some RCCL env variables
export FI_CXI_ATS=0
export HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_CROSS_NIC=1
export NCCL_SOCKET_IFNAME=hsn0
export CUDA_VISIBLE_DEVICES=7,6,5,4,3,2,1,0
export CUDA_DEVICE_MAX_CONNECTIONS=1
# AWS-OFI-RCCL
export LD_LIBRARY_PATH=${APP_ROOT}/repos/aws-ofi-rccl/lib:$LD_LIBRARY_PATH
# other
export MPICH_GPU_SUPPORT_ENABLED=1
export GPU_MAX_HW_QUEUES=1
export OFI_NCCL_USE_IPV6_TCP=1

# deepcam setup
export RUN_TAG="${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
BENCH_RCP_FIXED="\
    --gradient_accumulation_frequency 1 \
    --logging_frequency 10 \
    --save_frequency 0 \
    --seed $(date +%s) \
    --batchnorm_group_size 1 \
    --target_iou 0.80"
#BENCH_RCP_BASELINE_LR describes the learning rate for Baseline runs.
#It should not be modified.
BENCH_RCP_BASELINE_LR="\
    --start_lr 0.0055 \
    --lr_schedule type="multistep",milestones="800",decay_rate="0.1" \
    --lr_warmup_steps 400 \
    --lr_warmup_factor 1. \
    --weight_decay 1e-2 \
    --optimizer_betas 0.9 0.999"
BENCH_RCP_BASELINE="\
    ${BENCH_RCP_FIXED} \
    ${BENCH_RCP_BASELINE_LR}"

# define command
MAX_EPOCHS=1
cmd="srun --export=ALL --tasks-per-node=8 --gpus-per-node=8 \
        --gpu-bind=closest --gpus-per-task=1 \
        --cpu-bind=none --hint=nomultithread \
        python train.py \
            ${BENCH_RCP_BASELINE} \
            --data_dir_prefix ${APP_ROOT}/data/All-Hist \
            --run_tag ${RUN_TAG} \
            --output_dir ${JOB_OUTPUT_PATH} \
            --wireup_method nccl-slurm \
            --max_epochs ${MAX_EPOCHS} \
            --optimizer "Adam" \
            --local_batch_size 2"

# run with profiler
export WITH_PROFILER=1
OUTPUT_FILE="$JOB_OUTPUT_PATH/output-deepcam.log"
# clear cache
rm -rf ${MIOPEN_USER_DB_PATH}
mkdir -p ${MIOPEN_USER_DB_PATH}
# log start date
echo "start deepcam: $(date)" &>> $OUTPUT_FILE
# execute command
echo $cmd &>> $OUTPUT_FILE
eval $cmd &>> $OUTPUT_FILE
# log end date
echo "end deepcam: $(date)" &>> $OUTPUT_FILE

rm -rf ${MIOPEN_USER_DB_PATH}

# Run gpu benchmarks
COMM_TYPE=rccl
PERF_VARIABILITY_ROOT=/ccs/home/keshprad/perf-variability
echo running allreduce benchmark
bash $PERF_VARIABILITY_ROOT/gpu-benchmarks/allreduce/run_frontier.sh $COMM_TYPE $ROCM_VERSION $SLURM_JOB_NUM_NODES $JOB_OUTPUT_PATH
# echo running allgather benchmark
# bash $PERF_VARIABILITY_ROOT/gpu-benchmarks/allgather/run_frontier.sh $COMM_TYPE $ROCM_VERSION $SLURM_JOB_NUM_NODES $JOB_OUTPUT_PATH
echo running gemm benchmark
bash $PERF_VARIABILITY_ROOT/gpu-benchmarks/gemm/run_frontier.sh $ROCM_VERSION $SLURM_JOB_NUM_NODES $JOB_OUTPUT_PATH

echo "end run: $(date)"