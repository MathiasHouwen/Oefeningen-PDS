#!/bin/bash -l
#SBATCH --job-name=roundtrip_1node
#SBATCH --output=roundtrip_1node.out
#SBATCH --error=roundtrip_1node.err
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --partition=standard
#SBATCH --cluster=wice
#SBATCH --account=lp_h_pds_iiw
#SBATCH --partition=batch

module load OpenMPI/4.1.1-GCC-10.3.0
mpirun ./prog
