#!/bin/bash
for i in {1..30}
do
  sbatch scripts/bash/run_covasim_case_study.sh $i
done