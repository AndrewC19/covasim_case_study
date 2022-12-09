#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=8:00:00
#SBATCH --mem-per-cpu=4000
export SLURM_EXPORT_ENV=ALL
module load Anaconda3/5.3.0

# We assume that the conda environment 'venv' has already been created
source activate venv

# $1 is location.
# $2 is the variant. The variant used for each location is found under location_variants_seed_100.json.
# $3 is seed. Our experiments use a seed of 0.
# $4 is repeats. Our experiments use 30 repeats.
python data_collection.py --loc $1 --variant $2 --seed $3 --repeats $4
