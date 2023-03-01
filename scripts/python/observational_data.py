"""This script collates observational data collected for each location in Covasim."""
from utils import combine_results

LOCATION_OBSERVATIONAL_DATA_PATH = "results/sd_0.002"
OBSERVATIONAL_DATA_PATH = "data/observational_data.csv"


def combine_observational_data():
    df = combine_results(LOCATION_OBSERVATIONAL_DATA_PATH)
    df.to_csv(OBSERVATIONAL_DATA_PATH)


if __name__ == "__main__":
    combine_observational_data()
