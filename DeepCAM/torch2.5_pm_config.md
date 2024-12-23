
# Setup Guide for DeepCAM PyTorch Environment

## Load Required Modules

```bash
module load PrgEnv-nvidia cray-mpich cudatoolkit craype-accel-nvidia80 python/3.10 nccl cudnn cray-hdf5
```

## Create a Python Virtual Environment

```bash
python -m venv torch2.5-py3.10 --system-site-packages
source torch2.5-py3.10/bin/activate
pip install --upgrade pip
```

## Install PyTorch and Related Packages

```bash
pip install torch==2.5.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Install additional Python libraries required for the workflow.

```bash
pip install mpi4py pillow pandas pyyaml scipy h5py
pip install cupy-cuda12x
pip install pynvml
```

## Install MLPerf Libraries

```bash
pip install "git+https://github.com/NVIDIA/mlperf-common.git"
pip install "git+https://github.com/mlperf/logging.git"
```

## Install NVIDIA DALI


```bash
pip install nvidia-dali-cuda120
```

## Install NVIDIA Apex

```bash
git clone https://github.com/NVIDIA/apex
cd apex
git checkout 89cc215a
```

Install Apex dependencies.

```bash
pip install -r requirements.txt
```

Modify the `setup.py` file at line 32 as needed. Set the C++ compiler.

```bash
export CXX=/usr/bin/g++
```

Build and install Apex with CUDA extensions.

```bash
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

## Install DeepCAM PyTorch I/O Helpers

```bash
cd /pscratch/sd/c/cunyang/deepcam/optimized_pm/HPE+LBNL/benchmarks/deepcam/implementations/deepcam-pytorch/io_helpers
pip install -e .
```