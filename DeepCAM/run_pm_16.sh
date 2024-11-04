!/bin/bash
#SBATCH --job-name=test_job
#SBATCH --qos=regular
#SBATCH --constraint=gpu
#SBATCH --nodes=16
#SBATCH --time=01:00:00
#SBATCH --ntasks=64
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --output=%x-%j.out
#SBATCH -A m4641
module purge

#load the DeepCAM environment
#N10_DEEPCAM is defined in deepcam_env.sh
module load python
module load PrgEnv-nvidia
module load cudatoolkit
module load craype-accel-nvidia80
export CRAY_ACCEL_TARGET=nvidia80
module load python
module load cmake/3.24.3
export MPICH_GPU_SUPPORT_ENABLED=1

source ../deepcam_env.sh
#module load pytorch
#load hyperparameters
#BENCH_RCP is defined by bench_rcp.conf
#this reference convergence point (RCP) should not be modified
#source bench_rcp.conf

module load pytorch/2.1.0-cu12

BENCH_RCP_FIXED="\
    --gradient_accumulation_frequency 1 \
    --logging_frequency 10 \
    --save_frequency 0 \
    --min_epochs 1 \
    --seed $(date +%s) \
    --batchnorm_group_size 1 \
    --target_iou 0.80"

BENCH_RCP_OPTIMIZED="\
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

#BENCH_RCP_OPTIMIZED_LR describes the learning rate for Optimized runs.
#These hyperparameters may be modified.
BENCH_RCP_OPTIMIZED_LR="\
    --start_lr 0.0055 \
    --lr_schedule type="multistep",milestones="800",decay_rate="0.1" \
    --lr_warmup_steps 400 \
    --lr_warmup_factor 1. \
    --weight_decay 1e-2"

BENCH_RCP_BASELINE="\
    ${BENCH_RCP_FIXED} \
    ${BENCH_RCP_BASELINE_LR}"

BENCH_RCP_OPTIMIZED="\
    ${BENCH_RCP_OPTIMIZED} \
    ${BENCH_RCP_OPTIMIZED_LR}"


#the local batch size may be adjusted
#under the constraint that the global batch size is fixed to 2048,
#i.e. processes * local_batch_size = 2048.
#for example: local_batch_size=$(( 2048 / ${SLURM_NTASKS} ))
local_batch_size=2
module load nccl/2.19.4
export LD_LIBRARY_PATH=/global/common/software/nersc9/nccl/2.19.4/lib:$LD_LIBRARY_PATH

#other options within this script may be adjusted freely
data_dir=$N10_DEEPCAM/data/All-Hist
output_dir=output_dir
run_tag="${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
echo "go in to deepcam"

srun python3 $N10_DEEPCAM/baseline/src_deepCam/train.py \
     ${BENCH_RCP_BASELINE} \
     --wireup_method "nccl-slurm" \
    --run_tag ${run_tag} \
    --data_dir_prefix ${data_dir} \
    --output_dir ${output_dir} \
    --model_prefix "segmentation" \
    --optimizer "LAMB" \
    --max_epochs 1 \
    --max_inter_threads 1 \
    --local_batch_size ${local_batch_size} &> out_${SLURM_JOB_ID}_$(date +"%Y-%m-%d_%H-%M-%S").txt

#save results for successful run
if [[ $? == 0 ]]; then
   mkdir -p $N10_DEEPCAM/results/jobscripts
   mkdir -p $N10_DEEPCAM/results/logs
   if [[ $SLURM_JOB_QOS != interactive ]] && [[ SLURM_JOB_NAME != interactive ]]; then
       cp ${0} $N10_DEEPCAM/results/jobscripts/${SLURM_JOB_ID}.slurm
       cp -p $output_dir/logs/${run_tag}.log $N10_DEEPCAM/results/logs
   fi
fi
