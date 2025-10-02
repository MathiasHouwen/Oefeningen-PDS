#!/bin/bash -l
#SBATCH --job-name=cmake_test
#SBATCH --cluster=wice
#SBATCH --account=lp_h_pds_iiw
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=00:00:30
#SBATCH --error="%x.e%A"
#SBATCH --output="%x.o%A"

source /data/leuven/303/vsc30380/slurmhooks
module load CMake/3.20.1
cmake --version
sleep 22
