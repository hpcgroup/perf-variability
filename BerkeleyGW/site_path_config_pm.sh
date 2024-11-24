#!/bin/bash

#This file sets the paths for all BGW executable, library and input-data.
#It will need to be modified for each install site.
#To minimize the number of files that must be updated with this path data,
#all of the jobscripts will source this file.
#
#Jobscripts will still need to be updated to match
#a) your queue system, or
#b) your compute node configuration.

#N10_BGW=/path/to/berkeleygw-workflow
N10_BGW=/pscratch/sd/c/cunyang/bgw4/berkeleygw-workflow
if [[ -z "${N10_BGW}" ]]; then
    echo "The N10_BGW variable is not defined."
    echo "Please set N10_BGW in site_path_config.sh and try again."
    exit 0
fi

#libraries... you may need to add FFTW, or Scalapack or...
HDF_LIBPATH=/opt/cray/pe/hdf5-parallel/1.12.2.3/nvidia/20.7/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HDF_LIBPATH

#executables
BGW_DIR=$N10_BGW/BerkeleyGW-4.0/bin

#input data
Si_WFN_folder=$N10_BGW/Si_WFN_folder
Si214_WFN_folder=$Si_WFN_folder/Si214/WFN_file/
Si510_WFN_folder=$Si_WFN_folder/Si510/WFN_file/
Si998_WFN_folder=$Si_WFN_folder/Si998/WFN_file/
Si2742_WFN_folder=$Si_WFN_folder/Si2742/WFN_file/

#any modules that should be loaded at runtime
module swap PrgEnv-gnu PrgEnv-nvhpc