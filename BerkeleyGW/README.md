# BerkeleyGW 4.0 Installation Guide

## Prerequisites

Load the necessary modules:
```bash
module swap PrgEnv-gnu PrgEnv-nvhpc
module load cray-hdf5-parallel
module load cray-fftw
module load cray-libsci
module load python
```

## Cloning the Repository

Clone the BerkeleyGW 4.0 repository:
```bash
git clone https://github.com/pssg-int/BerkeleyGW-4.0
cd BerkeleyGW-4.0
```

## Configuration and Compilation

Copy the appropriate configuration file and compile the code:
```bash
cp config/perlmutter.nvhpc.gpu.nersc.gov.mk arch.mk
make -j cplx
```

## Data Download

For data download, please refer to [this link](https://gitlab.com/NERSC/N10-benchmarks/berkeleygw-workflow/-/tree/main/Si_WFN_folder).

## Further Information

For a more detailed introduction, please refer to [this link](https://gitlab.com/NERSC/N10-benchmarks/berkeleygw-workflow).

For the official manual, visit [BerkeleyGW 4.0 Manual](http://manual.berkeleygw.org/4.0/).