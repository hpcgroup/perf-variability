# nanoGPT README
For more detailed installation parameters, please refer to [nanoGPT install guide](https://github.com/axonn-ai/nanoGPT).

Repository: [AMG2023](https://github.com/hpcgroup/AMG2023/)


## Perlmutter Setup

### Setup steps

1. Clone the Repository
    ```sh
    git clone https://github.com/axonn-ai/nanoGPT.git
    cd nanoGPT
    ```

2.  Create Python Environment
    ```sh
    ./scripts/create_python_env_perlmutter.sh
    ```
    > Note: You may need to modify the path and torch version in `create_python_env_perlmutter.sh`.

3. Load PyTorch Module
    ```sh
    module load pytorch/2.0.1
    ```

4. Activate the Environment
    ```sh
    source path_to_nanogptENV/bin/activate
    ```

5. Download Data
    ```sh
    python nanoGPT/data/openwebtext/prepare.py
    ```

## Frontier Setup

### Setup steps

1. Clone the Repository
    ```sh
    git clone https://github.com/axonn-ai/nanoGPT.git
    cd nanoGPT
    ```

2.  Create Python Environment
    ```sh
    ./scripts/create_python_env_frontier.sh
    ```
    > Note: You may need to modify the WKSPC path and torch version in `create_python_env_frontier.sh`.

4. Activate the Environment
    ```sh
    source path_to_nanogptENV/bin/activate
    ```

5. Download Data
    ```sh
    python data/openwebtext/prepare.py
    ```