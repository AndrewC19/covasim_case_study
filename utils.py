import glob
import os
import random
import pandas as pd


def combine_results(results_dir: str):
    """Combine a directory of csv files into a single dataframe.

    :param results_dir: Path to the directory containing results CSVs.
    :return: A dataframe containing all data in the CSVs.
    """
    all_files = glob.glob(os.path.join(results_dir, "*.csv"))
    print(all_files)
    dfs = [pd.read_csv(csv_file, index_col=0) for csv_file in all_files]
    print(dfs)
    return pd.concat(dfs, ignore_index=True)


def less_data_overall(csv_path: str, proportion: float):
    random.seed(0)
    df = pd.read_csv(csv_path)
    return df.sample(frac=proportion)