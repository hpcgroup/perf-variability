#!/bin/bash
#SBATCH -t 00:30:00
#SBATCH --qos=regular
#SBATCH -N 64
#SBATCH --gpus-per-node=4
#SBATCH -A m4641
#SBATCH -J bgw_sig_Si510
#SBATCH  -C  gpu
#SBATCH -o BGW_SIGMA_%j.out
# #SBATCH --reservation=n10scaling
module purge

module load PrgEnv-gnu
module swap PrgEnv-gnu PrgEnv-nvhpc
module load cray-hdf5-parallel
module load cray-fftw
module load cray-libsci
module load python
module load cudatoolkit
module load craype-accel-nvidia80
export CRAY_ACCEL_TARGET=nvidia80
export MPICH_GPU_SUPPORT_ENABLED=1

source ../site_path_config.sh

mkdir BGW_SIGMA_$SLURM_JOB_ID
stripe_large BGW_SIGMA_$SLURM_JOB_ID
cd    BGW_SIGMA_$SLURM_JOB_ID
ln -s $BGW_DIR/sigma.cplx.x .
NNPOOL=8
cat ../sigma.inp |\
sed "s/NNPOOL/${NNPOOL}/g" > sigma.inp
ln -sfn  ${Si510_WFN_folder}/WFN_out.h5   ./WFN_inner.h5
ln -sfn  ${Si510_WFN_folder}/RHO          .
ln -sfn  ${Si510_WFN_folder}/VXC          .
ln -sfn  ${Si510_WFN_folder}/eps0mat.h5   .


ulimit -s unlimited
export OMP_PROC_BIND=true
export OMP_PLACES=threads
export HDF5_USE_FILE_LOCKING=FALSE
export BGW_HDF5_WRITE_REDIST=1
export BGW_WFN_HDF5_INDEPENDENT=1


export OMP_NUM_THREADS=16
srun -n 256 -c 32 --cpu-bind=cores --export=ALL,LD_PRELOAD=/pscratch/sd/c/cunyang/bgw4/software/lib/libmpiP.so  ./sigma.cplx.x