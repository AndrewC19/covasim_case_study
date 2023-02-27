#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=6:00:00
#SBATCH --mem-per-cpu=16000
export SLURM_EXPORT_ENV=ALL
module load Anaconda3/5.3.0

# We assume that the conda environment 'covenv' has already been created
source activate covenv

# $1 is the seed
# $2 is whether to run with less data or not
if [ $2 = true ] || [ $2 = "True" ] || [ $2 = 1 ]
then
  python scripts/python/covasim_case_study.py --seed $1 --ld
else
  python scripts/python/covasim_case_study.py --seed $1
fi