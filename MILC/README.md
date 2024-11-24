# MILC Installation Guide

## Prerequisites

Load the necessary modules:
```sh
module load PrgEnv-gnu
module load cudatoolkit
module load craype-accel-nvidia80
module load cmake/3.22.0
module unload xalt
export MPICH_GPU_SUPPORT_ENABLED=1
```

## Setup Download and Installation Paths

Define the paths for the build and installation:
```sh
BUILD_DIR=$(pwd)
MILC_QCD_DIR="${BUILD_DIR}/milc_qcd"
QUDA_DIR="${BUILD_DIR}/quda"
QUDA_BUILD="${BUILD_DIR}/quda/build"
QUDA_INSTALL_PREFIX="${BUILD_DIR}/quda_install"
PATH_TO_CUDA=$CUDA_HOME
PATH_TO_QUDA=$QUDA_INSTALL_PREFIX
```

## Clone and Build QUDA

Clone the QUDA repository and build it:
```sh
git clone --branch develop https://github.com/lattice/quda.git ${QUDA_DIR}
cd ${QUDA_DIR}

mkdir -p ${QUDA_BUILD}
cd ${QUDA_BUILD}

cmake \
    -G "Unix Makefiles" \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DQUDA_GPU_ARCH=sm_80 \
    -DQUDA_DIRAC_DEFAULT_OFF=ON \
    -DQUDA_DIRAC_STAGGERED=ON \
    -DQUDA_FORCE_HISQ=ON \
    -DQUDA_FORCE_GAUGE=ON \
    -DQUDA_MPI=ON \
    -DCMAKE_INSTALL_PREFIX=${QUDA_INSTALL_PREFIX} \
    ../

cmake --build . -- -j$(nproc)
cmake --build . -- install
```

## Clone and Build MILC

Clone the MILC repository and build it:
```sh
git clone --branch develop https://github.com/milc-qcd/milc_qcd.git ${MILC_QCD_DIR}
cd ${MILC_QCD_DIR}

cp Makefile ks_imp_rhmc
cd ks_imp_rhmc

export CUDA_MATH=/opt/nvidia/hpc_sdk/Linux_x86_64/22.5/math_libs/11.7/lib64

rm -f ../libraries/*.[ao]
make clean

MY_CC=cc \
MY_CXX=CC \
MPP=true \
OMP=true \
PRECISION=1 \
WANT_MIXED_PRECISION_GPU=0 \
CGEOM="-DFIX_NODE_GEOM -DFIX_IONODE_GEOM" \
CTIME="-DCGTIME -DFFTIME -DFLTIME -DGFTIME -DIOTIME" \
KSCGMULTI="-DKS_MULTICG=HYBRID -DMULTISOURCE" \
CUDA_HOME=${PATH_TO_CUDA} \
QUDA_HOME=${PATH_TO_QUDA} \
WANTQUDA=true \
WANT_FN_CG_GPU=true \
WANT_GF_GPU=true \
WANT_GA_GPU=true \
WANT_FL_GPU=true \
WANT_FF_GPU=true \
make -j 1 su3_rhmd_hisq
```

---

For downloading data, refer to [NERSC Lattice QCD Workflow](https://gitlab.com/NERSC/N10-benchmarks/lattice-qcd-workflow/-/tree/main/lattices).