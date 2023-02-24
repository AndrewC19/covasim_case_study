# Testing Causality in Scientific Modelling Software: Covasim Case Study

This repository contains the code for the Covasim case study from our paper entitled: "Testing Causality in Scientific Modelling Software."

In this case study, we apply the Causal Testing Framework (CTF) to the widely used open source COVID-19 agent-based modelling tool, Covasim.

Our goal here is to investigate whether the CTF can accurately estimate a series of statistical metamorphic
test (SMT) outcomes using causal knowledge and observational data. We also aim to identify whether accurate inferences can be achieved using 
small amounts of data.

### Repository contents
- `data/` contains all of the data collected for the case study. This data is used to create the figures.
  - `observational_data.csv` contains 4680 executions of Covasim (30 per location).
  - `observational_data_sample.csv` is a sample of the above data containing 187 executions.
  - `smt_data.csv` contains 9360 executions of Covasim: 30 executions per location (of which there are 156) per variant (of which there are two).
  - `smt_results.csv` contains the results of applying SMT to each location in Covasim.
  - `error_by_size.csv` contains the error (root-mean-square deviation, Spearman's rho, and Kendall's tau) corresponding to applications of the CTF to different amounts of data, ranging from 9 data points of the full data set to 4671 data points.
  - `error_by_size_first_500.csv` contains the same data as the above csv, but focuses on a smaller range of data points (more samples over a smaller range).
  - `location_variants_seed_0.json` maps each location to a COVID-19 variant and a randomly generated seed. These settings are used to produce `observational_data.csv`.
  - `location_fixed_variants_seed_0.json` maps each location to two COVID-19 variants (beta and alpha) and a pair of random seeds. These settings are used to produce `smt_results.csv`.
- `figures/` contains all of the figures that can be produced from this code.
- `scripts/` contains the various scripts used in data collection and analysis.
  - `bash/` contains bash scripts used to collect the data via HPC.
  - `python/` contains python scripts used to collect and analyse the data.
    - `data_collection.py` is used to collect both observational data and SMT data from Covasim.
    - `ctf_application.py` applies the CTF to our data.
    - `smt.py` applies SMT to the SMT data collected from Covasim.
    - `utils.py` contains various utils for cleaning and preparing the data.
    - `covasim_case_study.py` contains the code for analysing the collected data.
    - `subsets.py` contains the code for combining error data to form `error_by_size.csv`.
    - `observational_data.py` contains the code for combining observational data for each location into `observational_data.csv`.
- `dag.dot` is the Causal DAG for this case study.
- `requirements.txt` lists the requirements for this case study.
- `results/` is an empty directory that will be populated with observational data results during data collection.
- `fixed_results/` is an empty directory that will be populated with SMT data during data collection.

### Reproducing the Case Study
#### Installation
Begin by cloning this repository:
```
git clone https://github.com/AndrewC19/covasim_case_study
```

Change directory into the cloned directory:
```
cd covasim_case_study
```

Create and activate a fresh virtual environment using Python 3.9 (https://www.python.org/downloads/release/python-390/):
```
python3 -m venv case_study_venv
source case_study_venv/bin/activate 
```

Install requirements
```
pip install -r requirements.txt
```

In addition to these requirements, the CTF requires pygraphviz which requires a graphviz installation. The method for installing these requirements 
vary for different operating systems. Instructions can be found here: https://pygraphviz.github.io/documentation/stable/install.html
#### Data Analysis
To reproduce the figures in `figures/` from the data in `data/`, the following commands can be used.

1) To  apply the CTF to `data/observational_data.csv` and `data/observational_data_sample.csv` in order to produce `figures/causal_testing_framework.pdf` and `figures/less_data_causal_testing_framework.pdf`:
```
python scripts/python/covasim_case_study --ctf
``` 
2) To apply the location-specific regression model to `data/observational_data.csv` to produce `figures/location_regression.pdf`:
```
python scripts/python/covasim_case_study --loc
```
3) To plot the root-mean-square deviation (RMSD) against amount of data (`figures/rmsd_by_size.pdf`) using `data/error_by_size.csv`:
```
python scripts/python/covasim_case_study --rmsd
```
4) To repeat the above but focusing on fewer data points in more detail (`figures/rmsd_by_size_first_500.pdf`):
```
python scripts/python/covasim_case_study --rmsd --ld
```
5) To plot Spearman's rho against amount of data (`figures/spearmans_r_by_size.pdf`) using `data/error_by_size.csv`:
```
python scripts/python/covasim_case_study --src
```
6) To repeat the above but focusing on fewer data points in more detail (`figures/spearmans_r_by_size_first_500.pdf`):
```
python scripts/python/covasim_case_study --src --ld
```
7) To plot Kendall's tau against amount of data (`kendalls_t_by_size.pdf`) using `data/error_by_size.csv`: 
```
python scripts/python/covasim_case_study --krc
```
8) To repeat the above but focusing on fewer data points in more detail (`kendalls_t_by_size_first_500.pdf`):
```
python scripts/python/covasim_case_study --krc --ld
```

#### Data Collection
For this case study, data was collected using an HPC. The scripts used to collect the data can be found under `scripts/bash`. Nonetheless, here we explain how the python scripts called by these bash scripts (found under `scripts/python`) can be called to collect data.

##### To collect SMT data from Covasim:
SMT data is collected by running each location 30 times with two different variants: alpha and beta. This is achieved by running the python script:
```
python scripts/python/data_collection.py --loc $1 --variant $2 --seed $3 --repeats 30 --fixed 
```
Where `$1` is the location, `$2` is the variant (alpha or beta), and `$3` is the seed. For example, to get the SMT results for Australia:
```
python scripts/python/data_collection.py --loc australia --variant beta --seed 82239 --repeats 30 --fixed
python scripts/python/data_collection.py --loc australia --variant alpha --seed 346682 --repeats 30 --fixed
```
This will produce two files: `fixed_results/f_australia_seed_82239.csv` and `fixed_results/f_australia_seed_346682.csv`.
The bash script `scripts/bash/smt.sh` repeats this for every location. We then run `smt_data.csv` which produces `data/smt_data.csv` and `data/smt_results.csv`.
For reproducibility, we have included a json file (`data/location_fixed_variants_seed_0.json`) mapping each location to the seed used for the beta and alpha run, respectively.

##### To collect observational data from Covasim:
Observational data is collected by running each location 30 times with a randomly assigned dominant variant. Each execution randomly samples a slightly different version of this variant. This is achieved by running the python script:
```
python scripts/python/data_collection.py --loc $1 --variant $2 --seed $3 --repeats 30 --sd 0.002
```
Where `$1` is the location, `$2` is the variant, and `$3` is the seed. For example, to get the observational data for Australia:

```
python scripts/python/data_collection.py --loc australia --variant alpha --seed 783304 --repeats 30 --sd 0.002
```
This will produce the file `results/sd_0.002/australia_variant_alpha_seed_783304.csv`.
The bash script `scripts/bash/simulate_locations.sh` repeats this for every location. We then run `python scripts/python/observational_data.py` which produces `data/observational_data.csv`.
For reproducibility, we have included a json file (`data/location_variants_seed_0.json`) mapping each location to the variant and seed used for observational data collection.

##### To collect RMSD, Spearman's rho, and Kendall's tau on subsets of observational data:
We apply the CTF to increasingly smaller subsets of the full observational data set (`data/observational_data.csv`) and record the error 
in terms of the root-mean-square deviation (RMSD) and two measures of rank correlation (Spearman's rho and Kendall's tau).
This is achieved by running the following python command:
```
python scripts/python/covasim_case_study.py --seed $1
```
Where `$1` is the seed used for sampling the subsets.
This script will create 500 subsets of the observational data and apply the CTF to each one.
The subsets are saved under `results/subsets` and the results are saved in a file `results/data_size_seed_x.csv`, where `x` is the selected seed.
The bash script `scripts/bash/sample_data.sh`repeats this for seeds 1 to 30. We then run `python scripts/python/subsets.py` which combines these results into `data/error_by_size.csv`.

To reproduce the same data but focusing on fewer data points in greater detail, we can run the following python command:
```
python scripts/python/covasim_case_study.py --seed $1 --ld
```
Running `python scripts/python/subsets.py --ld` will then produce `data/error_by_size_first_500.csv`.

