#!/bin/bash
#SBATCH -N 16
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH --gpus-per-node=4
#SBATCH -J amg_16
#SBATCH --mail-user=cunyang@umd.edu
#SBATCH --mail-type=ALL
#SBATCH -t 00:10:00
#SBATCH -A m4641
#SBATCH --gpu-bind=none
module purge

module load PrgEnv-nvidia
module load cudatoolkit
module load craype-accel-nvidia80
export CRAY_ACCEL_TARGET=nvidia80
export MPICH_GPU_SUPPORT_ENABLED=1
export HYPRE_INSTALL_DIR=/pscratch/sd/c/cunyang/software


srun -n 64 -c 32 --export=ALL,LD_PRELOAD=/pscratch/sd/c/cunyang/software/lib/libmpiP.so ./amg -P 4 4 4 -n 128 64 64 -problem 1 -iter 500 &> out_${SLURM_JOB_ID}_$(date +"%Y-%m-%d_%H-%M-%S").txt