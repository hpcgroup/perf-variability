#!/bin/bash

# data directory
source $(dirname ${BASH_SOURCE[0]})/config_pm_common.sh

# hyperparameters
export LOCAL_BATCH_SIZE=2
export START_LR=0.004
export OPTIMIZER="LAMB" #"DistributedLAMB"
export LR_SCHEDULE_TYPE="multistep"
export LR_MILESTONES="800"
export LR_DECAY_RATE="0.1"
export LR_WARMUP_STEPS=100
export LR_WARMUP_FACTOR=1.
export WEIGHT_DECAY=0.01
export BATCHNORM_GROUP_SIZE=2
export MAX_EPOCHS=4

# data parameters
export SHUFFLE_MODE="global"
export DATA_FORMAT="dali-es-gpu"
export DATA_OVERSAMPLING_FACTOR=1
export PRECISION_MODE="amp"
export LOCAL_VALIDATION_BATCH_SIZE=1
export MAX_THREADS=8

# misc args
export ADDITIONAL_ARGS="${ADDITIONAL_ARGS} --disable_comm_overlap --enable_graph --enable_groupbn"
#export ADDITIONAL_ARGS="${ADDITIONAL_ARGS} --enable_jit --disable_comm_overlap --enable_graph --enable_groupbn"

# system parameters
export DGXNNODES=64
WALLTIME_MINUTES=7
export WALLTIME=$(( 15 + (${NEXP} * ${WALLTIME_MINUTES}) ))
#export SBATCH_ARRAY_INX="1-5%1"