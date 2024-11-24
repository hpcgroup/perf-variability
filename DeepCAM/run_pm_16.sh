#!/bin/bash

CONFIG=${1:-configs/config_pm_16x4x1_npy.sh}
echo "Using config ${CONFIG}"
source ${CONFIG}
unset ${!SLURM_@};
sbatch -N $DGXNNODES -t $WALLTIME  ${EXTRA_SBATCH_ARGS} run_pm.sub