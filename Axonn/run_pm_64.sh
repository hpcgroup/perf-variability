#!/bin/bash
#SBATCH -t 00:30:00
#SBATCH --qos=regular
#SBATCH --gpus-per-node=4
#SBATCH -A m4641_g
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=4
#SBATCH --constraint=gpu
#SBATCH --exclusive

export HF_HOME="/pscratch/sd/c/cunyang/saved_data"
export HF_TRANSFORMERS_CACHE="${HF_HOME}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
module load cuda
module load python
module load cudatoolkit
module load pytorch/2.3.1
#. $SCRATCH/axonn_venv/bin/activate
. /global/common/software/m4641/venv-2.3.1/bin/activate

NNODES=$SLURM_JOB_NUM_NODES
GPUS=$(( NNODES * 4 ))
## master addr and port
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export WORLD_SIZE=${GPUS}

## nccl env vars to speedup stuff
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NET_GDR_LEVEL=PHB
export CUDA_VISIBLE_DEVICES=3,2,1,0
export NCCL_CROSS_NIC=1
export NCCL_SOCKET_IFNAME=hsn
export NCCL_NET="AWS Libfabric"
export FI_CXI_RDZV_THRESHOLD=0
export FI_CXI_RDZV_GET_MIN=0
export FI_CXI_OFLOW_BUF_SIZE=1073741824
export FI_CXI_OFLOW_BUF_COUNT=1
export MPICH_GPU_SUPPORT_ENABLED=0

#MODEL_ID="meta-llama/Meta-Llama-3-8B"
#MODEL_ID="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
MODEL_ID="meta-llama/Llama-2-7b-hf"
#MODEL_ID="deepseek-ai/DeepSeek-V2-Lite"
#MODEL_ID="Qwen/Qwen1.5-MoE-A2.7B"
#MODEL_ID="allenai/OLMoE-1B-7B-0924"
STRATEGY=${1:-axonn}


ARG_FILE="sample_args_file.json"

SCRIPT="finetune_pl.py --file $ARG_FILE"

#WANDB_ARGS="--wandb-log \
#       --wandb-project lightning-axonn-new-easy-api \
#       --wandb-run-name ${MODEL_ID}-${STRATEGY}-gpus-${GPUS}-asym-index-cpu"


run_cmd="srun -C gpu -N $NNODES -n $GPUS -c 32  --cpu-bind=cores --gpus-per-node=4 ./get_rank.sh python -u $SCRIPT &> out_${SLURM_JOB_ID}_$(date +"%Y-%m-%d_%H-%M-%S").txt "

echo $run_cmd
eval $run_cmd
