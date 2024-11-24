#!/bin/bash
#SBATCH -N 16
#SBATCH -t 00:20:00
#SBATCH -C gpu
#SBATCH -A m4641
#SBATCH --job-name=L6496-gen
#SBATCH -o L6496-gen-gpu.o%j
#SBATCH -q regular
#SBATCH --cpus-per-task=32
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none
#SBATCH --exclusive

module purge

module load PrgEnv-nvidia
export CRAY_ACCEL_TARGET=nvidia80

module load PrgEnv-gnu
module load cudatoolkit
module load craype-accel-nvidia80
module unload xalt
export MPICH_GPU_SUPPORT_ENABLED=1


N10_MILC="../../.."
MILC_QCD_DIR=${N10_MILC}/build/PM-quda/milc_qcd
LATTICE_DIR=${N10_MILC}/lattices

if [ ! -d lattices ]; then
    ln -s $LATTICE_DIR ./lattices
fi

#bind=${N10_MILC}/bin/bind4-perlmutter.sh
exe=${MILC_QCD_DIR}/ks_imp_rhmc/su3_rhmd_hisq
input=input_6496

export OMP_NUM_THREADS=16
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

export QUDA_ENABLE_GDR=1
export QUDA_MILC_HISQ_RECONSTRUCT=13
export QUDA_MILC_HISQ_RECONSTRUCT_SLOPPY=9

# Tuning results are stored in qudatune_dir.
qudatune_dir="$PWD/qudatune"
export QUDA_RESOURCE_PATH=${qudatune_dir}
if [ ! -d ${qudatune_dir} ]; then
    mkdir ${qudatune_dir}
fi

command="srun -n $SLURM_NTASKS \
     -c $SLURM_CPUS_PER_TASK \
      --export=ALL,LD_PRELOAD=/pscratch/sd/c/cunyang/software/lib/libmpiP.so  $exe $input out_${SLURM_JOB_ID}_$(date +"%Y-%m-%d_%H-%M-%S").txt"

echo $command

$command