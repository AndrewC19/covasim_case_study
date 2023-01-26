#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=2:00:00
#SBATCH --mem-per-cpu=16000
export SLURM_EXPORT_ENV=ALL
module load Anaconda3/5.3.0

# We assume that the conda environment 'covenv' has already been created
source activate covenv

# $1 is location.
# $2 is the variant. The variant used for each location is found under location_variants_seed_0.json.
# $3 is seed. This seed used for each location is found under location_variants_seed_0.json.
# $4 is repeats. Our experiments use 30 repeats.
# $5 is whether to use a fixed beta. This is used for producing highly uniform data.
# $6 is the standard deviation of the distributions from which beta is drawn. This is used when fixed beta is false.
if [ $5 = true ] || [ $5 = "True" ] || [ $5 = 1 ]
then
  python data_collection.py --loc $1 --variant $2 --seed $3 --repeats $4 --fixed
else
  python data_collection.py --loc $1 --variant $2 --seed $3 --repeats $4 --sd $6
fi