"""This script applies SMT to the collected SMT data to obtain gold standard
estimates for increasing beta from 0.016 to 0.02672 in each location in
Covasim."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import combine_results

SMT_CSV_DATA_PATH = "data/smt_data.csv"
SMT_RESULTS_PATH = "fixed_results"


def combine_smt_results():
    smt_df = combine_results(SMT_RESULTS_PATH)
    smt_df.to_csv(SMT_CSV_DATA_PATH)


def smt_all_locations():
    smt_df = pd.read_csv(SMT_CSV_DATA_PATH)
    results_dict = {}
    for location in smt_df["location"].unique():
        change, confidence_intervals = change_variant_in_location(location)
        results_dict[location] = {"change_in_infections": change,
                                  "low_ci": confidence_intervals[0],
                                  "high_ci": confidence_intervals[1],
                                  "confidence_level": 0.95}
    pd.DataFrame.from_dict(results_dict,
                           orient='index').to_csv("data/smt_results.csv")


def plot_change_in_infections_by_location():
    smt_results_df = pd.read_csv("smt_results.csv", index_col=[0])
    smt_results_df.sort_index(inplace=True)
    for i, (location, row) in enumerate(smt_results_df.iterrows()):
        mean_change_in_infections = smt_results_df.at[location,
                                                      "change_in_infections"]
        low_ci = smt_results_df.at[location, "low_ci"]
        high_ci = smt_results_df.at[location, "high_ci"]

        low_err = abs(mean_change_in_infections - low_ci)
        high_err = abs(mean_change_in_infections - high_ci)

        print(mean_change_in_infections, low_ci, high_ci)
        plt.errorbar(i, smt_results_df.at[location, "change_in_infections"],
                     yerr=np.array(low_err, high_err),
                     ls=':')
        # plt.scatter(i, smt_results_df.at[location, "change_in_infections"])
    plt.tick_params(bottom=False, labelbottom=False)
    plt.ylabel("Change in cumulative infections")
    plt.xlabel("Locations")
    plt.tight_layout()
    plt.show()


def change_variant_in_location(location: str):
    smt_df = pd.read_csv(SMT_CSV_DATA_PATH)
    location_smt_df = smt_df.loc[smt_df["location"] == location]

    # Split data into source and follow-up based on beta setting
    source_df = location_smt_df[location_smt_df["beta"] == 0.016]
    follow_up_df = location_smt_df[location_smt_df["beta"] == 0.02672]

    # Obtain data for cumulative infections
    source_infections = source_df["cum_infections"]
    follow_up_infections = follow_up_df["cum_infections"]

    # Compute the mean change in infections that results from the change in beta
    mean_source_infections = source_df["cum_infections"].mean()
    mean_follow_up_infections = follow_up_df["cum_infections"].mean()
    mean_change_in_infections = (mean_follow_up_infections -
                                 mean_source_infections)

    # Bootstrap 95% confidence intervals
    ci_low, ci_high = smt_bootstrap_confidence_intervals(source_infections,
                                                         follow_up_infections,
                                                         0.95,
                                                         2000)
    return mean_change_in_infections, [ci_low, ci_high]


def smt_bootstrap_confidence_intervals(source_data,
                                       follow_up_data,
                                       confidence_level=0.95,
                                       n_resamples=100):
    """Bootstrap confidence intervals using the empirical bootstrap method.

    The specific method implemented here is based on the following course notes:
    https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2014/resources/mit18_05s14_reading24/

    :param source_data: An iterable collection of data for the source test case.
    :param follow_up_data: An iterable collection of data for the follow-up test
    case.
    :param confidence_level: Confidence level of bootstrapped confidence
    intervals.
    :param n_resamples: Number of times to resample the original sample.
    """
    source_data = np.array(source_data)
    follow_up_data = np.array(follow_up_data)
    sample_mean_diff = follow_up_data.mean() - source_data.mean()

    # Resample the data n_resamples to form a matrix of samples with n_resamples
    # rows. Note that each sample (row) is the same size as the original.
    source_bootstrap_samples_indices = np.random.randint(
        0, len(source_data), size=(n_resamples, len(source_data)))
    source_bootstrap_samples = np.array(
        [source_data[i] for i in source_bootstrap_samples_indices])

    follow_up_bootstrap_samples_indices = np.random.randint(
        0, len(follow_up_data), size=(n_resamples, len(follow_up_data)))
    follow_up_bootstrap_samples = np.array(
        [follow_up_data[i] for i in follow_up_bootstrap_samples_indices])

    # Compute the difference in mean between each source, follow-up sample pair
    bootstrap_sample_mean_diffs = [
        follow_up_bootstrap_samples[sample].mean() -
        source_bootstrap_samples[sample].mean() for sample in range(n_resamples)
    ]

    # Subtract the difference in means of the original samples and sort in
    # ascending order
    bootstrap_sample_deltas = [
        bootstrap_sample_mean_diff - sample_mean_diff for
        bootstrap_sample_mean_diff in bootstrap_sample_mean_diffs
    ]
    bootstrap_sample_deltas.sort()

    # Obtain upper and lower percentile sample means (corrected for 0 index)
    lower_percentile_index = round(confidence_level * n_resamples) - 1
    lower_percentile = bootstrap_sample_deltas[lower_percentile_index]
    upper_percentile_index = round((1 - confidence_level) * n_resamples) - 1
    upper_percentile = bootstrap_sample_deltas[upper_percentile_index]

    # Subtract lower and upper percentile from mean to obtain lower and upper
    # confidence intervals
    return [sample_mean_diff - lower_percentile,
            sample_mean_diff - upper_percentile]


if __name__ == "__main__":
    combine_smt_results()
    smt_all_locations()
