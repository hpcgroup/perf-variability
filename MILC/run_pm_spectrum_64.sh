#!/bin/bash
#SBATCH -N 64
#SBATCH -C gpu
#SBATCH -t 00:30:00
#SBATCH -A m4641
#SBATCH --job-name=L144288-spe
#SBATCH -o L144288-spe-gpu.o%j
#SBATCH -q regular
#SBATCH --cpus-per-task=32
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
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

bind=${N10_MILC}/bin/bind4-perlmutter.sh
exe=${MILC_QCD_DIR}/ks_spectrum/ks_spectrum_hisq
input=input_144288
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
     -c 32 \
     $bind $exe $input out_64_${SLURM_JOB_ID}_$(date +"%Y-%m-%d_%H-%M-%S").txt"

echo $command

$command
