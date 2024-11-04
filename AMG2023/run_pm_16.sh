#!/bin/bash
#SBATCH -N 16
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH --gpus-per-node=4
#SBATCH -J amg_test
#SBATCH --mail-user=cunyang@umd.edu
#SBATCH --mail-type=ALL
#SBATCH -t 00:30:00
#SBATCH -A m4641
#SBATCH --gpu-bind=none
module purge

module load PrgEnv-nvidia
module load cudatoolkit
module load craype-accel-nvidia80
export CRAY_ACCEL_TARGET=nvidia80
export MPICH_GPU_SUPPORT_ENABLED=1
export HYPRE_INSTALL_DIR=/pscratch/sd/c/cunyang/software


srun -n 64 -c 32 ./amg -P 4 4 4 -n 100 100 100 -problem 1 &> out_${SLURM_JOB_ID}_$(date +"%Y-%m-%d_%H-%M-%S").txt
