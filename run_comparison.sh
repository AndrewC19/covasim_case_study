#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=6:00:00
#SBATCH --mem-per-cpu=16000
export SLURM_EXPORT_ENV=ALL
module load Anaconda3/5.3.0

# We assume that the conda environment 'covenv' has already been created
source activate covenv

# $1 is the seed
python comparison.py --seed $1