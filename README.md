# GPU Supercomputing Performance Variability Study

This repository is dedicated to researching performance variability on modern GPU supercomputers. We perform continuous testing on the Perlmutter and Frontier systems using various HPC workloads, including AMG2023, MILC, BerkeleyGW, DeepCAM, and AxoNN, on 16-node and 64-node configurations. Our study utilizes profiling tools to collect and analyze data related to network, I/O, and so on.

## Repository Structure

Each directory in this repository contains:
- Slurm scripts for running tests (16 nodes and 64 nodes) on Perlmutter and Frontier.
- Build instructions and references for compiling the workloads on these systems.

## System Architectures

### Perlmutter
Perlmutter is a HPE Cray EX supercomputer.
Each GPU node contains:
- **CPU**: AMD EPYC 7763 (Milan)
- **GPU**: 4x NVIDIA A100 GPUs

### Frontier
Frontier is a HPE Cray EX Supercomputer.
- **CPU**:  AMD Epyc 7713 (Trento)
- **GPU**: 4x AMD MI250X Instinct GPUs. The AMD MI250X has two Graphic Compute Dies (GCDs) per module. This gives a total of 8 GCDs per node

## Workload Descriptions

### DeepCAM Overview
The DeepCAM benchmark trains a deep learning model to identify extreme weather phenomena, such as tropical cyclones and atmospheric rivers, in CAM5 climate simulation data. The N10 DeepCAM benchmark is based on the MLPerf HPC DeepCAM benchmark, derived from the 2018 ACM Gordon Bell prize-winning work "Exascale Deep Learning for Climate Analytics." It is a semantic segmentation application using high-resolution (768x1152) scientific images with 16 channels, compared to the 3-channel images typical in commercial cases. The MLPerf HPC DeepCAM reference implementation is architecture-agnostic, with the N10 version serving as the baseline implementation with minor modifications.

### MILC - Lattice QCD Simulation
The Lattice QCD workflow consists of "generation" and "spectrum" stages. During the generation stage, lattices are propagated until an equilibrium distribution is reached, using checkpoint-restart methods to manage long equilibration times. The spectrum stage samples equilibrated lattices to measure physical properties. The NERSC-10 Lattice QCD benchmark employs MILC’s implementation of SU(3) lattice gauge theory, using the Rational Hybrid Monte Carlo (RHMC) algorithm for generation and the ks_spectrum_hisq application for spectrum analysis. MILC is written in C and parallelized using MPI, with support for specialized libraries like QPhiX and QUDA for optimized performance on CPUs and GPUs.

### BerkeleyGW - Material Property Prediction
BerkeleyGW simulates optical properties of materials and nanostructures, essential for energy conversion and electronic devices. It processes data from DFT-based codes and focuses on the Epsilon and Sigma modules to compute dielectric functions and electronic self-energy. The benchmark uses the General Plasmon Pole approximation and excludes DFT, Kernel, and Absorption stages. BerkeleyGW, written in Fortran and C/C++, leverages MPI, OpenMP, and GPU-specific constructs like OpenACC.

### AMG2023 - Algebraic Multigrid Solver
AMG2023 is a parallel algebraic multigrid solver for linear systems on unstructured grids, using hypre-2.27.0 or higher. It employs MPI for parallelism and OpenMP for threading, using a simple data decomposition strategy in 3D grids. AMG2023 is optimized for NVIDIA, AMD, and Intel GPUs through CUDA, HIP, and SYCL.

### AxoNN - Neural Network Training Framework
AxoNN is a parallel framework for training deep neural networks. It is designed to efficiently scale across large numbers of GPUs and uses advanced communication strategies to optimize performance.


