#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=2:00:00
#SBATCH --mem-per-cpu=16000
export SLURM_EXPORT_ENV=ALL
module load Anaconda3/5.3.0

# We assume that the conda environment 'covenv' has already been created
source activate covvenv

# $1 is location.
# $2 is the variant. The variant used for each location is found under location_variants_seed_0.json.
# $3 is seed. This seed used for each location is found under location_variants_seed_0.json.
# $4 is repeats. Our experiments use 30 repeats.
python data_collection.py --loc $1 --variant $2 --seed $3 --repeats $4
