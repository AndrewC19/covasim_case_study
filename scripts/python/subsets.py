"""This script collates subset error data collected."""
import argparse
from utils import combine_results

SUBSET_SEEDS_DATA_PATH = "results/subsets"


def combine_subset_data(out_file_name):
    df = combine_results(SUBSET_SEEDS_DATA_PATH)
    df = df.sort_values(by=['data_points'])
    df.to_csv(out_file_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ld", action="store_true", help="Whether to append `first_500` to file name.")
    args = parser.parse_args()
    if args.ld:
        combine_subset_data("data/error_by_size_first_500.csv")
    else:
        combine_subset_data("data/error_by_size.csv")
