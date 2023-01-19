"""This script applies SMT to the collected SMT data to obtain gold standard
estimates for increasing beta from 0.016 to 0.02672 in each location in
Covasim."""

import pandas as pd
import numpy as np

SMT_CSV_DATA_PATH = "fixed_results/complete_data/smt.csv"


def double_beta_in_location(location: str):
    smt_df = pd.read_csv(SMT_CSV_DATA_PATH)
    location_smt_df = smt_df.loc[smt_df["location"] == location]
    source_df = location_smt_df[location_smt_df["beta"] == 0.016]
    follow_up_df = location_smt_df[location_smt_df["beta"] == 0.02672]
    source_infections = source_df["cum_infections"].mean()
    follow_up_infections = follow_up_df["cum_infections"].mean()
    change_in_infections = follow_up_infections - source_infections
    ci_low, ci_high = bootstrap_confidence_intervals()
    return change_in_infections


def bootstrap_confidence_intervals(data,
                                   confidence_level=0.95,
                                   n_resamples=100):
    """Bootstrap confidence intervals using the empirical bootstrap method.

    The specific method implemented here is based on the following course notes:
    https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2014/resources/mit18_05s14_reading24/

    :param data: An iterable collection of data.
    :param confidence_level: Confidence level of bootstrapped confidence
    intervals.
    :param n_resamples: Number of times to resample the original sample.
    """
    data = np.array(data)
    sample_mean = data.mean()

    # Resample the data n_resamples to form a matrix of samples with n_resamples
    # rows. Note that each sample (row) is the same size as the original.
    bootstrap_samples_indices = np.random.randint(0, len(data),
                                                  size=(n_resamples, len(data)))
    bootstrap_samples = np.array([data[i] for i in bootstrap_samples_indices])

    # Compute the mean of each sample (row), subtract the sample mean, and sort
    # in ascending order
    bootstrap_sample_means = [sample.mean() for sample in bootstrap_samples]
    bootstrap_sample_mean_diffs = [bs_sample_mean - sample_mean for
                                   bs_sample_mean in bootstrap_sample_means]
    bootstrap_sample_mean_diffs.sort()

    # Obtain upper and lower percentile sample means (corrected for 0 index)
    lower_percentile_index = round(confidence_level * n_resamples) - 1
    lower_percentile = bootstrap_sample_mean_diffs[lower_percentile_index]
    upper_percentile_index = round((1 - confidence_level) * n_resamples) - 1
    upper_percentile = bootstrap_sample_mean_diffs[upper_percentile_index]

    # Subtract lower and upper percentile from mean to obtain lower and upper
    # confidence intervals
    return [sample_mean - lower_percentile, sample_mean - upper_percentile]


if __name__ == "__main__":
    confidence_intervals = bootstrap_confidence_intervals(
        [100, 200, 300],
        0.9,
        20)
    print(confidence_intervals)
