#!/bin/bash

source ~/programs/activate_conda
conda activate rakali_env_3.13
export VERNO=25.5
module load nvhpc
module load misc/nvhpc-build/${VERNO}/netcdf-c/
module load misc/nvhpc-build/${VERNO}/netcdf-fortran/
module load hdf5
export PYTHON=python3
