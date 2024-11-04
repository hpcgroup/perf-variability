#!/bin/bash
#SBATCH -t 00:30:00
#SBATCH --qos=regular
#SBATCH -N 16
#SBATCH --gpus-per-node=4
#SBATCH -A m4641
#SBATCH -J bgw_eps_Si510
#SBATCH -C gpu
#SBATCH -o BGW_EPSILON_%j.out
#SBATCH --exclusive
# #SBATCH --reservation=n10scaling
module purge

module load PrgEnv-gnu
module swap PrgEnv-gnu PrgEnv-nvhpc
module load cray-hdf5-parallel
module load cray-fftw
module load cray-libsci
module load cudatoolkit
module load craype-accel-nvidia80
export CRAY_ACCEL_TARGET=nvidia80
export MPICH_GPU_SUPPORT_ENABLED=1

source ../site_path_config.sh

mkdir BGW_EPSILON_$SLURM_JOBID
stripe_large BGW_EPSILON_$SLURM_JOBID
cd    BGW_EPSILON_$SLURM_JOBID
ln -s $BGW_DIR/epsilon.cplx.x .
ln -s  ../epsilon.inp .
ln -sfn  ${Si510_WFN_folder}/WFNq.h5      .
ln -sfn  ${Si510_WFN_folder}/WFN_out.h5   ./WFN.h5


ulimit -s unlimited
export OMP_PROC_BIND=true
export OMP_PLACES=threads
export HDF5_USE_FILE_LOCKING=FALSE
export BGW_HDF5_WRITE_REDIST=1
export BGW_WFN_HDF5_INDEPENDENT=1


export OMP_NUM_THREADS=16
srun -n 64 -c 32 --cpu-bind=cores ./epsilon.cplx.x &> out_${SLURM_JOB_ID}_$(date +"%Y-%m-%d_%H-%M-%S").txt
