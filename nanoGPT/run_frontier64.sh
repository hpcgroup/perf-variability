#!/bin/bash
#SBATCH -N 64
#SBATCH -n 512
#SBATCH -q normal
#SBATCH -J nanogpt
#SBATCH -t 01:00:00
#SBATCH -A csc569
#SBATCH --output /lustre/orion/csc569/scratch/keshprad/perfvar/nanoGPT_logs/64nodes/%x-%j/job-output.log
#SBATCH --error /lustre/orion/csc569/scratch/keshprad/perfvar/nanoGPT_logs/64nodes/%x-%j/job-error.log
#SBATCH --exclusive
# Run like: sbatch run_frontier64.sh

export JOB_OUTPUT_PATH=/lustre/orion/csc569/scratch/keshprad/perfvar/nanoGPT_logs/64nodes/$SLURM_JOB_NAME-$SLURM_JOB_ID
OUTPUT_FILE=$JOB_OUTPUT_PATH/output-nanoGPT.log
ERROR_FILE=$JOB_OUTPUT_PATH/error-nanoGPT.log

# Run gpu benchmarks
COMM_TYPE=rccl
PERF_VARIABILITY_ROOT=/ccs/home/keshprad/perf-variability
echo running allreduce benchmark
bash $PERF_VARIABILITY_ROOT/gpu-benchmarks/allreduce/run_frontier.sh $COMM_TYPE $SLURM_JOB_NUM_NODES $JOB_OUTPUT_PATH
# echo running allgather benchmark
# bash $PERF_VARIABILITY_ROOT/gpu-benchmarks/allgather/run_frontier.sh $COMM_TYPE $SLURM_JOB_NUM_NODES $JOB_OUTPUT_PATH
echo running gemm benchmark
bash $PERF_VARIABILITY_ROOT/gpu-benchmarks/gemm/run_frontier.sh $SLURM_JOB_NUM_NODES $JOB_OUTPUT_PATH

APP_ROOT=/lustre/orion/csc569/scratch/keshprad/nanoGPT
cd $APP_ROOT

export SCRATCH="/lustre/orion/csc569/scratch/keshprad"
export WRKSPC="${SCRATCH}/nanoGPT"
export HF_HOME="${SCRATCH}/.cache/hf"
export HF_TRANSFORMERS_CACHE="${HF_HOME}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
cd $WRKSPC

# load modules
rocm_version=6.1.3
module reset
module load PrgEnv-gnu/8.5.0
module load rocm/${rocm_version}
module load craype-accel-amd-gfx90a
module load cray-python/3.9.13.1
module load gcc-native/12.3
module load cray-mpich/8.1.30
# activate env
source ${WRKSPC}/axonn_nanogpt/bin/activate

NNODES=$SLURM_JOB_NUM_NODES
GPUS=$(( NNODES * 8 ))
## master addr and port
export MASTER_ADDR=$(hostname -i)
export MASTER_PORT=3442
export WORLD_SIZE=${GPUS}

## nccl env vars to speedup stuff
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NET_GDR_LEVEL=PHB
export CUDA_VISIBLE_DEVICES=7,6,5,4,3,2,1,0
export NCCL_CROSS_NIC=1
export NCCL_SOCKET_IFNAME=hsn0
export NCCL_NET="AWS Libfabric"
export NCCL_TIMEOUT=1200
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1200
export MPICH_GPU_SUPPORT_ENABLED=0
# AWS-OFI-RCCL
export LD_LIBRARY_PATH="${WRKSPC}/repos/aws-ofi-rccl/lib:$LD_LIBRARY_PATH"

SCRIPT="train_frontier.py config/train_gpt_neox_20B.py"

# run without profiler
export WITH_PROFILER=0
# log start date
echo start nanoGPT_withoutprof: $(date)
run_cmd="srun -N $NNODES -n $GPUS --cpu-bind=cores --gpus-per-node=8 --ntasks-per-node=8 scripts/get_rank.sh python -u $SCRIPT &> $JOB_OUTPUT_PATH/output-nanoGPT_withoutprof.log"
echo $run_cmd
eval $run_cmd
# log end date
echo end nanoGPT_withoutprof: $(date)


# run with profiler
export WITH_PROFILER=1
# log start date
echo start nanoGPT: $(date)
run_cmd="srun -N $NNODES -n $GPUS --cpu-bind=cores --gpus-per-node=8 --ntasks-per-node=8 scripts/get_rank.sh python -u $SCRIPT &> $JOB_OUTPUT_PATH/output-nanoGPT.log"
echo $run_cmd
eval $run_cmd
# log end date
echo end nanoGPT: $(date)
