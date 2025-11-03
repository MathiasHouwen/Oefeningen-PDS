#!/bin/bash -l
#SBATCH --cluster=wice
#SBATCH --account=lp_h_pds_iiw
#SBATCH --job-name=oef21
#SBATCH --output=oef21.out
#SBATCH --error=oef21.err
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=00:10:00
module --purge
module load CUDA/12.1.1
nvcc Oef21.cu -o cudaprog
./cudaprog