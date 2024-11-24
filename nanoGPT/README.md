# nanoGPT Setup Instructions

## Clone the Repository

```sh
git clone https://github.com/axonn-ai/nanoGPT.git
```

## Create Python Environment

```sh
./scripts/create_python_env_perlmutter.sh
```

> Note: You may need to modify the path and torch version in `create_python_env_perlmutter.sh`.

## Load PyTorch Module

```sh
module load pytorch/2.0.1
```

## Activate the Environment

```sh
source path_to_nanogptENV/bin/activate
```

## Download Data

```sh
python nanoGPT/data/openwebtext/prepare.py
```