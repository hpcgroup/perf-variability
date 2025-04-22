# DeepCAM README
For more detailed installation parameters, please refer to DeepCAM install guide

Perlmutter Repository: [hpc_results_v3.0](https://github.com/hpcgroup/hpc_results_v3.0)  
Frontier Repository: [hpc](https://github.com/hpcgroup/hpc)


## Perlmutter Setup

### Setup steps

## Frontier Setup

### Setup steps

#### 1. Pytorch Install
- Load modules
    ```bash
    module reset
    module load PrgEnv-gnu/8.5.0
    module load rocm/6.1.3
    module load craype-accel-amd-gfx90a
    module load cray-python/3.9.13.1
- Create env variables
    ```bash
    DEEPCAM_ROOT=/lustre/orion/csc547/scratch/keshprad/deepcam/
    PYVENV_ROOT=${DEEPCAM_ROOT}/.venv
    PYVENV_SITEPKGS=${PYVENV_ROOT}/lib/python3.9/site-packages

    cd ${DEEPCAM_ROOT}
    ```
- Create python virtual env
    ```bash
    python -m venv ${PYVENV_ROOT}
    source ${PYVENV_ROOT}/bin/activate
    ```
- Install torch and mpi4py
    ```bash
    # torch==2.5.0
    pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/rocm6.1

    MPICC="cc -shared" pip install --no-cache-dir --no-binary=mpi4py mpi4py
    ```
- Install AWS-OCI-RCCL plugin
    ```bash
    mkdir -p ${DEEPCAM_ROOT}/repos
    cd ${DEEPCAM_ROOT}/repos

    rocm_version=6.1.3
    # Load modules
    module load PrgEnv-gnu/8.5.0
    module load rocm/$rocm_version
    module load craype-accel-amd-gfx90a
    module load gcc-native/12.3
    module load cray-mpich/8.1.30
    #module load libtool
    libfabric_path=/opt/cray/libfabric/1.15.2.0

    # Download the plugin repo
    git clone --recursive https://github.com/ROCmSoftwarePlatform/aws-ofi-rccl
    cd aws-ofi-rccl

    # Build the plugin
    ./autogen.sh
    export LD_LIBRARY_PATH=/opt/rocm-$rocm_version/hip/lib:$LD_LIBRARY_PATH
    PLUG_PREFIX=$PWD

    CC=hipcc CFLAGS=-I/opt/rocm-$rocm_version/rccl/include ./configure \
    --with-libfabric=$libfabric_path --with-rccl=/opt/rocm-$rocm_version --enable-trace \
    --prefix=$PLUG_PREFIX --with-hip=/opt/rocm-$rocm_version/hip --with-mpi=$MPICH_DIR

    make
    make install

    # Reminder to export the plugin to your path
    echo $PLUG_PREFIX
    echo "Add the following line in the environment to use the AWS OFI RCCL plugin"
    echo "export LD_LIBRARY_PATH="$PLUG_PREFIX"/lib:$""LD_LIBRARY_PATH"
    ```
- Install supporting dependencies
    ```bash
    cd ${DEEPCAM_ROOT}

    pip install wandb
    pip install gym
    pip install pyspark
    pip install scikit-learn
    pip install scikit-image
    pip install opencv-python
    pip install wheel
    pip install tomli
    pip install h5py

    # tensorboard
    pip install tensorboard
    pip install tensorboard_plugin_profile
    pip install tensorboard-plugin-wit
    pip install tensorboard-pytorch

    pip install git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git
    ```
- Install mlperf-logging
    ```bash
    mkdir -p ${DEEPCAM_ROOT}/repos
    cd ${DEEPCAM_ROOT}/repos

    git clone -b hpc-1.0-branch https://github.com/mlcommons/logging mlperf-logging
    # may need to manually change mlperf-logging/VERSION to a valid version number (e.g. 1.0.0.rc2)
    pip install -e mlperf-logging

    rm ${PYVENV_SITEPKGS}/mlperf-logging.egg-link
    cp -r ./mlperf-logging/mlperf_logging ${PYVENV_SITEPKGS}/mlperf_logging
    cp -r ./mlperf-logging/mlperf_logging.egg-info ${PYVENV_SITEPKGS}/mlperf_logging.egg-info
    ```

#### 2. Download src code
- Download from PSSG Frontier repo for DeepCAM (linked at top of README)
    ```bash
    # REPLACE WITH YOUR PATH
    PRFX=/lustre/orion/csc547/scratch/keshprad
    DEEPCAM_ROOT=${PRFX}/deepcam

    mkdir -p ${DEEPCAM_ROOT}
    cd ${DEEPCAM_ROOT}

    git clone https://github.com/hpcgroup/hpc.git hpc
    ```

#### 3. Download dataset with globus
- [Globus Link](https://app.globus.org/file-manager?origin_id=0b226e2c-4de0-11ea-971a-021304b0cca7&origin_path=%2F)
    - Download to `$DEEPCAM_ROOT/data`