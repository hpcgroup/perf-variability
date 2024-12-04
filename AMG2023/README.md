# AMG2023 README
For more detailed installation parameters, please refer to the [installation document](https://github.com/pssg-int/AMG2023/blob/main/amg-doc.pdf).

## Perlmutter Compilation

Repository: [AMG2023](https://github.com/pssg-int/AMG2023)

### Steps to Compile

1. Clone the repository:
    ```sh
    cd AMG2023
    git clone -b v2.27.0 https://github.com/hypre-space/hypre.git
    ```

2. Navigate to the source directory:
    ```sh
    cd hypre/src
    ```

3. Load necessary modules:
    ```sh
    module load cudatoolkit/11.7
    module load gcc/10.3.0
    module load cray-mpich cray-libsci
    ```

4. Configure the build:
    ```sh
    ./configure --with-cuda --enable-device-memory-pool --enable-mixedint --prefix=/pscratch/sd/c/cunyang/AMG2023 --with-gpu-arch=80
    ```

5. Load additional modules:
    ```sh
    module load cmake/3.24.3
    module load PrgEnv-cray
    module load cudatoolkit/11.7
    module load craype-accel-nvidia80
    ```

6. Create and navigate to the build directory:
    ```sh
    cd ../../AMG2023
    mkdir build
    cd build
    ```

7. Run CMake to configure the build system:
    ```sh
    cmake -DHYPRE_PREFIX=/pscratch/sd/c/cunyang/AMG2023 ..
    ```

## Frontier Installation
1. Load modules
    ```sh
    module reset

    module load cray-mpich/8.1.28
    module load craype-accel-amd-gfx90a
    module load rocm
    export MPICH_GPU_SUPPORT_ENABLED=1

    # load compatible cmake version
    module load Core/24.07
    module load cmake/3.27.9
    ```
2. Configure hypre
    - Clone hypre v2.27.0 and navigate to src: 
        ```sh
        git clone -b v2.27.0 https://github.com/hypre-space/hypre.git
        cd into ~/hypre/src
        ```
    - Configure hypre (in hypre/src)
        ```sh
        ./configure --with-hip --with-gpu-arch=gfx90a --with-MPI-lib-dirs="${MPICH_DIR}/lib" --with-MPI-libs="mpi" --with-MPI-include="${MPICH_DIR}/include"
        ```
    - Compile hypre (in hypre/src)
        ```sh
        # build with make
        make
        ```
3. Configure AMG2023
    - Clone repo: 
        ```sh
        git clone https://github.com/pssg-int/AMG2023`
        cd AMG2023
        ```
    - Configure cmake
        ```sh
        mkdir build && cd build

        cmake .. -DHYPRE_PREFIX=/ccs/home/keshprad/hypre/src/hypre/ -DCMAKE_EXE_LINKER_FLAGS="-lrocsparse -lrocrand"
        ```
    - Compile AMG2023 (in AMG2023/build)
        ```sh
        make install
        ```
