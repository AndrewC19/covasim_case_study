import pandas as pd
import numpy as np
import random
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import argparse
from time import time
from scipy.stats import spearmanr, kendalltau
from pathlib import Path
from matplotlib import rcParams
from matplotlib.pyplot import figure
from ctf_application import increasing_beta

# REQUIRES LATEX INSTALLATION: UNCOMMENT TO PRODUCE FIGURES USING LATEX FONTS
# rc_fonts = {
#     "font.family": "serif",
#     'font.serif': 'Linux Libertine O',
#     'font.size': 14,
#     "text.usetex": True
# }
# rcParams.update(rc_fonts)
figure(figsize=(14, 5), dpi=150)


def gold_standard_results(df):
    """Read in gold standard SMT results and return as a dict mapping true effect of increasing beta from 0.016 to
    0.02672 to each location."""
    gold_standard_dict = {}
    for location in df.index:
        gold_standard_dict[location] = df.at[location, "change_in_infections"]
    sorted_gold_standard_dict = {k: v for k, v in
                                 sorted(gold_standard_dict.items())}
    return sorted_gold_standard_dict


def naive_regression(df):
    """Apply a naive regression in which the cumulative infections are regressed against beta."""
    naive_regression_eqn = "cum_infections ~ np.log(beta) + np.power(np.log(beta), 2)"
    individuals = pd.DataFrame(index=['source', 'follow-up'],
                               data={'beta': [0.016, 0.02672]})
    locations = df["location"].unique()
    results_dict = {}
    for location in locations:
        naive_model = smf.ols(naive_regression_eqn, data=df).fit()
        predicted_outcomes = naive_model.predict(individuals)
        ate = predicted_outcomes['follow-up'] - predicted_outcomes['source']
        results_dict[location] = ate
    sorted_results_dict = {k: v for k, v in sorted(results_dict.items())}
    return sorted_results_dict


def location_regression(df):
    """Apply a naive regression in which the cumulative infections are regressed against beta for each location
    separately. This is achieved by including an interaction term between beta and location."""
    naive_regression_eqn = "cum_infections ~ np.log(beta) + np.power(np.log(beta), 2) + C(location) + " \
                           "np.log(beta):C(location)"
    individuals = pd.DataFrame(index=['source', 'follow-up'],
                               data={'beta': [0.016, 0.02672]})
    locations = df["location"].unique()
    results_dict = {}
    for location in locations:
        naive_model = smf.ols(naive_regression_eqn, data=df).fit()
        individuals["location"] = location
        predicted_outcomes = naive_model.predict(individuals)
        ate = predicted_outcomes['follow-up'] - predicted_outcomes['source']
        results_dict[location] = ate
    sorted_results_dict = {k: v for k, v in sorted(results_dict.items())}
    return sorted_results_dict


# def causal_regression(df):
#     """Perform a causal regression, adjusting for work, school and household contacts, and average relative
#     susceptibility."""
#     # causal_regression_eqn = """cum_infections ~ beta + np.power(beta, 2) +
#     #                            avg_contacts_s + avg_contacts_w + avg_rel_sus +
#     #                            avg_contacts_h + beta:avg_contacts_s +
#     #                            beta:avg_contacts_w + beta:avg_contacts_h +
#     #                            beta:avg_rel_sus
#     #                        """
#
#     # PROCESS TO CONSTRUCT THIS REGRESSION EQUATION:
#     # First, we used our causal dag to identify sufficient adjustment sets.
#     # We then used our domain knowledge to select, from these adjustment sets,
#     # which is most appropriate. In this case, using total contacts in workplace
#     # and school, and avg rel sus is better than age because it's the shape of
#     # the age distribution that matters, not the mean.
#     # We also add quadratic terms where variables are known or suspected to have
#     # a curvilinear relationship with cumulative infections. This may not be
#     # strictly necessary after linearising the data, but is there for safety.
#
#     # causal_regression_eqn = """cum_infections ~ np.log(beta) + np.power(np.log(beta), 2) +
#     #                                np.log(avg_contacts_h) + np.power(np.log(avg_contacts_h), 2) +
#     #                                np.log(avg_rel_sus) + np.power(np.log(avg_rel_sus), 2) +
#     #                                np.log(total_contacts_w) + np.power(np.log(total_contacts_w), 2) +
#     #                                np.log(total_contacts_s) + np.power(np.log(total_contacts_s), 2) +
#     #                                np.log(beta):np.log(total_contacts_w) +
#     #                                np.log(beta):np.log(total_contacts_s) +
#     #                                np.log(beta):np.log(avg_contacts_h) +
#     #                                np.log(beta):np.log(avg_rel_sus)
#     #                            """
#     to_log = ["beta", "avg_contacts_h", "avg_rel_sus", "total_contacts_w", "total_contacts_s"]
#     logged_df = df.copy()
#     logged_df[to_log] = np.log(logged_df[to_log])
#     causal_regression_eqn = """cum_infections ~ beta + np.power(beta, 2) +
#                                        avg_contacts_h + np.power(avg_contacts_h, 2) +
#                                        avg_rel_sus + np.power(avg_rel_sus, 2) +
#                                        total_contacts_w + np.power(total_contacts_w, 2) +
#                                        total_contacts_s + np.power(total_contacts_s, 2) +
#                                        beta:total_contacts_w +
#                                        beta:total_contacts_s +
#                                        beta:avg_contacts_h +
#                                        beta:avg_rel_sus
#                                    """
#     causal_model = smf.ols(causal_regression_eqn, data=logged_df).fit()
#     source_follow_up_dict = {'beta': [np.log(0.016), np.log(0.02672)]}
#     # adjustment_set = ["avg_contacts_h", "avg_contacts_w", "avg_contacts_s",
#     #                   "avg_rel_sus"]
#     adjustment_set = ["avg_contacts_h", "avg_age", "avg_rel_sus", "total_contacts_s",
#                       "total_contacts_w"]
#     locations = df["location"].unique()
#     results_dict = {}
#     for location in locations:
#         adjustment_dict = {}
#         location_df = logged_df.loc[logged_df["location"] == location]
#         for adjustment_var in adjustment_set:
#             adjustment_dict[adjustment_var] = location_df[adjustment_var].mean()
#         location_individuals = pd.DataFrame(
#             index=['source', 'follow-up'],
#             data=source_follow_up_dict | adjustment_dict
#         )
#         predicted_outcomes = causal_model.predict(location_individuals)
#         ate = predicted_outcomes['follow-up'] - predicted_outcomes['source']
#         results_dict[location] = ate
#     sorted_results_dict = {k: v for k, v in sorted(results_dict.items())}
#     return sorted_results_dict


def rmsd(true_values, estimates):
    """Calculate the root-mean-square deviation (error)."""
    difference = np.array(true_values) - np.array(estimates)
    squared_difference = np.power(difference, 2)
    sum_squared_difference = np.sum(squared_difference)
    normalised_sum_squared_difference = sum_squared_difference / len(estimates)
    rmsds = np.sqrt(normalised_sum_squared_difference)
    return rmsds


def rmsd_from_dicts(true_dict, estimate_dict):
    """Given a dict for true location effects and another for estimated location effects, return the overall
    root-mean-square deviation (error)."""
    true_values = []
    estimates = []
    for location in true_dict.keys():
        true_values.append(true_dict[location])
        estimates.append(estimate_dict[location])
    return rmsd(true_values, estimates)


def rmspe(true_values, estimates):
    """Calculate the root mean square percentage error."""
    difference = np.array(true_values) - np.array(estimates)
    percentage_difference = np.divide(difference, true_values)
    squared_percentage_difference = np.power(percentage_difference, 2)
    sum_squared_percentage_difference = np.sum(squared_percentage_difference)
    mspes = sum_squared_percentage_difference / len(estimates)
    rmspes = np.sqrt(mspes)
    return rmspes


def rmspe_from_dicts(true_dict, estimate_dict):
    """Given a dict for true location effects and another for estimated location effects, return the overall
    root-mean-square-percentage error (error)."""
    true_values = []
    estimates = []
    for location in true_dict.keys():
        true_values.append(true_dict[location])
        estimates.append(estimate_dict[location])
    return rmspe(true_values, estimates)


def individual_errors(true_dict, estimate_dict):
    """Compute the difference between the true and estimated effects for each location."""
    errors = {}
    for location in true_dict.keys():
        errors[location] = estimate_dict[location] - true_dict[location]
    return errors


def plot_estimates(gold_standard, naive_estimates, causal_estimates, color='blue', label='Causal Testing Framework',
                   title=None):
    """Plot the estimates on a scatter plot, showing location vs. effect for each estimate and the gold standard."""
    figure(figsize=(14, 5), dpi=150)
    ascending_gold_standard = {k: v for k, v in sorted(gold_standard.items(),
                                                       key=lambda item: item[1]
                                                       )}

    xs = list(ascending_gold_standard.keys())
    naive_estimate = min(naive_estimates.values())

    # Naive estimate should be the same for all locations as it provides one estimate for all locations.
    # Where data is incomplete, we can therefore use any other location estimate as they are the same.
    # This assertion checks this assumption.
    assert naive_estimate == max(naive_estimates.values())

    ys_true = [ascending_gold_standard[location] for location in xs]
    ys_naive = [naive_estimates[location] if location in naive_estimates else naive_estimate for location in xs]
    ys_causal = [causal_estimates[location] for location in xs]

    locations = xs.copy()
    capitalised_locations = [loc.capitalize() for loc in locations]
    plt.scatter(capitalised_locations, ys_true, label="Gold Standard", marker='.', s=8, color='green')
    plt.scatter(capitalised_locations, ys_naive, label="Standard Regression", marker='.', s=8, color='red')
    plt.scatter(capitalised_locations, ys_causal, label=label, marker='.', s=8, color=color)
    plt.xticks(rotation=60, ha='right', fontsize=6)
    plt.xlim(-1, len(capitalised_locations))
    plt.ylabel("Change in Cumulative Infections")
    if title:
        plt.title(title)
    plt.plot()
    plt.legend()
    plt.tight_layout()
    out_pdf = label.replace(" ", "_").lower() + ".pdf"
    plt.savefig(f"figures/{out_pdf}", format="pdf", dpi=150)
    plt.show()


def rmsd_vs_data(rand_seed: int, n_samples: int, less_data: bool):
    """Obtain the RMSD and RMPSE (error) and rank correlation vs. the amount of data. This experiment repeatedly applies
    the CTF to smaller subsets of the original data and calculates:
    (1) The error as the root-mean-square deviation over all location estimates.
    (2) The rank correlation using two metrics: Spearman's Rho and Kendall's Tau.

    The rank correlation captures the extent to which the ordering of the (ascending) magnitude of the effects is
    preserved.

    If less_data is specified, the samples are focused over a range of 500 data points.
    """
    results_dict = {"data_points": [],
                    "rmsd": [],
                    "rmpse": [],
                    "spearmans_r": [],
                    "spearmans_p_val": [],
                    "kendalls_tau": [],
                    "kendalls_p_val": []}
    gold_standard_df = pd.read_csv("data/smt_results.csv", index_col=0)
    gold_standard_ates = gold_standard_results(gold_standard_df)
    sorted_gold_standard_ates = {k: v for k, v in sorted(gold_standard_ates.items(), key=lambda item: item[1])}

    # Order gold standard location effects in ascending order
    gold_standard_locations_by_ascending_effect = list(sorted_gold_standard_ates.keys())

    # Map each location to a number starting with the location with the smallest observed effect
    ranks_dict = {location: i for i, location in enumerate(gold_standard_locations_by_ascending_effect)}
    gold_standard_ranks = list(ranks_dict.values())

    # Given the random seed, sample smaller subsets of the original data
    sample_data("data/observational_data.csv", n_samples, rand_seed, less_data)

    # For each sample, apply the CTF and compute the RMSD and Spearman's rank correlation to the gold standard
    for x in range(1, n_samples + 1):
        if less_data:
            data_size = x / 4680  # 4680 is the total number of data points in the full data set.
        else:
            data_size = x / n_samples
        data = pd.read_csv(f"results/subsets/size_{data_size}/seed_{rand_seed}.csv")
        ctf_estimates = increasing_beta(f"results/subsets/size_{data_size}/seed_{rand_seed}.csv")

        # List CTF estimates in ascending order
        sorted_ctf_estimates = {k: v for k, v in sorted(ctf_estimates.items(), key=lambda item: item[1])}
        ctf_locations_by_ascending_effect = list(sorted_ctf_estimates)

        # Using the gold standard mapping, obtain a list of ranks ready for the CTF for Spearman's rank correlation
        # and Kendall's Tau
        ctf_ranks = [ranks_dict[location] for location in ctf_locations_by_ascending_effect]

        # Write results to dict and save
        spearmans = spearmanr(gold_standard_ranks, ctf_ranks)
        kendalls = kendalltau(gold_standard_ranks, ctf_ranks)
        results_dict["spearmans_r"].append(spearmans.correlation)
        results_dict["spearmans_p_val"].append(spearmans.pvalue)
        results_dict["kendalls_tau"].append(kendalls.correlation)
        results_dict["kendalls_p_val"].append(kendalls.pvalue)
        results_dict["data_points"].append(len(data))
        results_dict["rmsd"].append(rmsd_from_dicts(gold_standard_ates, ctf_estimates))
        results_dict["rmpse"].append(rmspe_from_dicts(gold_standard_ates, ctf_estimates))

    # Write results to CSV
    df = pd.DataFrame(results_dict)
    df.to_csv(f"results/subsets/error_by_data_size_seed_{rand_seed}.csv")


def plot_rmsd_vs_data(rmsd_csv_path, output_name):
    """Plot RMSD vs. amount of data from CSV."""
    df = pd.read_csv(rmsd_csv_path)
    xs = df['data_points'].unique()
    ys_mean = []
    ys_min = []
    ys_max = []
    ys_upper = []
    ys_lower = []
    for x in xs:
        x_df = df.loc[df['data_points'] == x]['rmsd']
        mean = x_df.mean()
        ci = 1.96*(x_df.std()/np.sqrt(len(x_df)))
        ys_mean.append(mean)
        ys_lower.append(mean - ci)
        ys_upper.append(mean + ci)

        # ys_min.append(x_df.min())
        # ys_max.append(x_df.max())
    plt.xlabel("Data Points")
    plt.ylabel("RMSD")
    plt.plot(xs, ys_mean, color='red', linewidth=.8)
    plt.xscale("log")
    plt.xlim(df['data_points'].min(), df['data_points'].max())
    plt.fill_between(xs, ys_lower, ys_upper, color='red', alpha=.2)
    plt.tight_layout()
    plt.savefig(f"figures/{output_name}.pdf", format='pdf', dpi=150)
    plt.show()


def plot_spearmans_r_vs_data(spearmans_csv_path, output_name):
    """Plot Spearman's rho vs. amount of data from CSV."""
    df = pd.read_csv(spearmans_csv_path)
    xs = df['data_points'].unique()
    ys_mean = []
    ys_lower = []
    ys_upper = []
    for x in xs:
        x_df = df.loc[df['data_points'] == x]['spearmans_r']
        mean = x_df.mean()
        ci = 1.96 * (x_df.std() / np.sqrt(len(x_df)))
        ys_mean.append(mean)
        ys_lower.append(mean - ci)
        ys_upper.append(mean + ci)
    plt.xlabel("Data Points")
    plt.ylabel(r"Spearman's $\rho$")
    plt.plot(xs, ys_mean, color='blue', linewidth=.8)
    plt.xscale("log")
    plt.xlim(df['data_points'].min(), df['data_points'].max())
    plt.ylim(0, 1)
    plt.fill_between(xs, ys_lower, ys_upper, color='blue', alpha=.2)
    plt.tight_layout()
    plt.savefig(f"figures/{output_name}.pdf", format='pdf', dpi=150)
    plt.show()


def plot_kendalls_tau_vs_data(kendalls_csv_path, output_name):
    """Plot Kendall's tau vs. amount of data from CSV."""
    df = pd.read_csv(kendalls_csv_path)
    xs = df['data_points'].unique()
    ys_mean = []
    ys_lower = []
    ys_upper = []
    for x in xs:
        x_df = df.loc[df['data_points'] == x]['kendalls_tau']
        mean = x_df.mean()
        ci = 1.96 * (x_df.std() / np.sqrt(len(x_df)))
        ys_mean.append(mean)
        ys_lower.append(mean - ci)
        ys_upper.append(mean + ci)
    plt.xlabel("Data Points")
    plt.ylabel(r"Kendall's $\tau$")
    plt.plot(xs, ys_mean, color='green', linewidth=.8)
    plt.xscale("log")
    plt.xlim(df['data_points'].min(), df['data_points'].max())
    plt.ylim(0, 1)
    plt.fill_between(xs, ys_lower, ys_upper, color='green', alpha=.2)
    plt.tight_layout()
    plt.savefig(f"figures/{output_name}.pdf", format='pdf', dpi=150)
    plt.show()


def sample_data(data_path: str, n_samples: int, rand_seed: int, less_data: bool):
    """Take uniform samples of the data to obtain subsets of the original data of increasing size using a random seed.

    If the less_data parameter is set to true, the samples are taken over the range of 500 data points."""
    random.seed(rand_seed)
    for data_size in range(1, n_samples + 1):
        rand_state = random.randint(0, 100000)
        if less_data:
            data_size_frac = data_size / 4680  # 4680 is the total number of data points in the full data set.
        else:
            data_size_frac = data_size / n_samples
        df = pd.read_csv(data_path)
        smaller_df = df.sample(frac=data_size_frac, random_state=rand_state)
        out_path = f"results/subsets/size_{data_size_frac}/"
        path = Path(out_path)
        path.mkdir(parents=True, exist_ok=True)
        out_csv = path / f"seed_{rand_seed}.csv"
        smaller_df.to_csv(out_csv)


def ctf_results():
    observational_df = pd.read_csv("data/observational_data.csv",
                                   index_col=0)
    gold_standard_df = pd.read_csv("data/smt_results.csv", index_col=0)
    gold_standard_ates = gold_standard_results(gold_standard_df)
    sorted_gold_standard_ates = {k: v for k, v in sorted(gold_standard_ates.items(), key=lambda item: item[1])}

    # Order gold standard location effects in ascending order
    gold_standard_locations_by_ascending_effect = list(sorted_gold_standard_ates.keys())

    # Map each location to a number starting with the location with the smallest observed effect
    ranks_dict = {location: i for i, location in enumerate(gold_standard_locations_by_ascending_effect)}
    gold_standard_ranks = list(ranks_dict.values())

    naive_ates = naive_regression(observational_df)
    ctf_start_time = time()
    ctf_estimates = increasing_beta("data/observational_data.csv")
    ctf_end_time = time()

    ctf_estimates_df = pd.DataFrame.from_dict(ctf_estimates, orient='index')
    ctf_estimates_df = ctf_estimates_df.rename(columns={0: "estimate"})
    ctf_estimates_df.index.name = "location"
    ctf_estimates_df.to_csv("data/ctf_estimates.csv")
    print(ctf_estimates)
    sorted_ctf_estimates = {k: v for k, v in sorted(ctf_estimates.items(), key=lambda item: item[1])}
    ctf_locations_by_ascending_effect = list(sorted_ctf_estimates)

    # Using the gold standard mapping, obtain a list of ranks ready for the CTF for Spearman's rank correlation
    # and Kendall's Tau
    ctf_ranks = [ranks_dict[location] for location in ctf_locations_by_ascending_effect]
    kendalls = kendalltau(gold_standard_ranks, ctf_ranks)
    print(f"Kendall's Tau: {kendalls.correlation}")
    print(f"p-value: {kendalls.pvalue}")

    # Get the RMSD
    ctf_rmsd = rmsd_from_dicts(gold_standard_ates, ctf_estimates)
    ctf_rmspe = rmspe_from_dicts(gold_standard_ates, ctf_estimates)
    naive_regression_rmsd = rmsd_from_dicts(gold_standard_ates, naive_ates)
    naive_regression_rmspe = rmspe_from_dicts(gold_standard_ates, naive_ates)
    print(f"CTF RMSD: {ctf_rmsd}")
    print(f"CTF RMPSE: {ctf_rmspe}")
    print(f"Naive regression RMSD: {naive_regression_rmsd}")
    print(f"Naive regression RMPSE: {naive_regression_rmspe}")

    plot_estimates(gold_standard_ates, naive_ates, ctf_estimates, title="Results using 4680 data points")
    print(f"CTF run time: {ctf_end_time - ctf_start_time}")


def less_data_ctf_results():
    observational_df = pd.read_csv("data/observational_data_sample.csv",  # TODO: Replace with updated sample.
                                   index_col=0)
    gold_standard_df = pd.read_csv("data/smt_results.csv", index_col=0)
    gold_standard_ates = gold_standard_results(gold_standard_df)
    naive_ates = naive_regression(observational_df)
    ctf_estimates = increasing_beta("data/observational_data_sample.csv")
    plot_estimates(gold_standard_ates, naive_ates, ctf_estimates, label="Less data Causal Testing Framework",
                   title="Results using 187 data points")


def location_results():
    observational_df = pd.read_csv("data/observational_data.csv",
                                   index_col=0)
    gold_standard_df = pd.read_csv("data/smt_results.csv", index_col=0)
    gold_standard_ates = gold_standard_results(gold_standard_df)
    sorted_gold_standard_ates = {k: v for k, v in sorted(gold_standard_ates.items(), key=lambda item: item[1])}

    # Order gold standard location effects in ascending order
    gold_standard_locations_by_ascending_effect = list(sorted_gold_standard_ates.keys())

    # Map each location to a number starting with the location with the smallest observed effect
    ranks_dict = {location: i for i, location in enumerate(gold_standard_locations_by_ascending_effect)}
    gold_standard_ranks = list(ranks_dict.values())

    naive_ates = naive_regression(observational_df)
    location_ates = location_regression(observational_df)

    sorted_location_estimates = {k: v for k, v in sorted(location_ates.items(), key=lambda item: item[1])}
    location_estimates_locations_by_ascending_effect = list(sorted_location_estimates)

    location_ranks = [ranks_dict[location] for location in location_estimates_locations_by_ascending_effect]
    kendalls = kendalltau(gold_standard_ranks, location_ranks)
    print(f"Kendall's Tau: {kendalls.correlation}")
    print(f"p-value: {kendalls.pvalue}")

    # Get the RMSD
    location_rmsd = rmsd_from_dicts(gold_standard_ates, location_ates)
    print(f"Location RMSD: {location_rmsd}")
    plot_estimates(gold_standard_ates, naive_ates, location_ates, label="Location Regression", color="black")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ctf", action="store_true", help="Reproduce CTF results.")
    parser.add_argument("--loc", action="store_true", help="Reproduce location regression results.")
    parser.add_argument("--rmsd", action="store_true", help="Plot RMSD vs. amount of data.")
    parser.add_argument("--src", action="store_true", help="Plot Spearman's rank correlation vs. amount of data.")
    parser.add_argument("--krc", action="store_true", help="Plot Kendall's rank correlation vs. amount of data.")
    parser.add_argument("--seed", required=False, help="Seed for sampling subsets of data and calculating RMSD and rank"
                                                       " correlation of CTF results vs. the gold standard.")
    parser.add_argument("--ns", type=int, help="Number of samples to take if seed parameter is specified.",
                        default=500)
    parser.add_argument("--ld", action="store_true", help="Collect subsets of data focusing on less data points."
                                                          " Specifically, the specified number of samples (--ns) are"
                                                          " collected over the first 500 data points.",
                        default=False)
    args = parser.parse_args()
    if args.ctf:
        ctf_results()
        less_data_ctf_results()
    if args.loc:
        location_results()
    if args.rmsd:
        if args.ld:
            plot_rmsd_vs_data("data/error_by_size_first_500.csv", "rmsd_by_size_first_500")
        else:
            plot_rmsd_vs_data("data/error_by_size.csv", "rmsd_by_size")
    if args.src:
        if args.ld:
            plot_spearmans_r_vs_data("data/error_by_size_first_500.csv", "spearmans_r_by_size_first_500")
        else:
            plot_spearmans_r_vs_data("data/error_by_size.csv", "spearmans_r_by_size")
    if args.krc:
        if args.ld:
            plot_kendalls_tau_vs_data("data/error_by_size_first_500.csv", "kendalls_t_by_size_first_500")
        else:
            plot_kendalls_tau_vs_data("data/error_by_size.csv", "kendalls_t_by_size")
    if args.seed:
        rmsd_vs_data(args.seed, args.ns, args.ld)
