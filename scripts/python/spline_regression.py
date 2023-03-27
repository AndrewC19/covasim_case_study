import pandas as pd
import numpy as np
import random
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import covasim as cv
from time import time
from scipy.stats import spearmanr, kendalltau
from pathlib import Path
from matplotlib import rcParams
from matplotlib.pyplot import figure
from ctf_application import increasing_beta

# REQUIRES LATEX INSTALLATION: UNCOMMENT TO PRODUCE FIGURES USING LATEX FONTS
rc_fonts = {
    "font.family": "serif",
    'font.serif': 'Linux Libertine O',
    'font.size': 14,
    "text.usetex": True
}
rcParams.update(rc_fonts)
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


def causal_regression(df):
    """Perform a causal regression, adjusting for work, school and household contacts, and average relative
    susceptibility."""
    causal_regression_eqn = """
                cum_infections ~
                beta + bs(beta, degree=3, df=5) + 
                np.log(avg_rel_sus) + np.power(np.log(avg_rel_sus), 2) +
                np.log(total_contacts_w) + np.power(np.log(total_contacts_w), 2) +
                np.log(total_contacts_s) + np.power(np.log(total_contacts_s), 2) +
                np.log(total_contacts_h) + np.power(np.log(total_contacts_h), 2) +
                beta:np.log(total_contacts_w) +
                beta:np.log(total_contacts_s) +
                beta:np.log(total_contacts_h) +
                beta:np.log(avg_rel_sus)
        """
    causal_model = smf.ols(causal_regression_eqn, data=df).fit()
    print(causal_model.summary())
    source_follow_up_dict = {'beta': [0.016, 0.02672]}
    adjustment_set = ["total_contacts_h", "avg_rel_sus", "total_contacts_s",
                      "total_contacts_w"]
    results_dict = {}
    ref_df = pd.read_csv("data/observational_data.csv")
    locations = ref_df["location"].unique()
    for location in locations:
        adjustment_dict = {}
        location_df = ref_df.loc[ref_df["location"] == location]
        for adjustment_var in adjustment_set:
            adjustment_dict[adjustment_var] = location_df[adjustment_var].mean()
        location_individuals = pd.DataFrame(
            index=['source', 'follow-up'],
            data=source_follow_up_dict | adjustment_dict
        )
        predicted_outcomes = causal_model.predict(location_individuals)
        ate = predicted_outcomes['follow-up'] - predicted_outcomes['source']
        results_dict[location] = ate
    sorted_results_dict = {k: v for k, v in sorted(results_dict.items())}
    return sorted_results_dict


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
    plt.ylim(min(ys_true)*0.8, max(ys_true)*1.2)
    plt.ylabel("Change in Cumulative Infections")
    if title:
        plt.title(title)
    plt.plot()
    plt.legend()
    plt.tight_layout()
    out_pdf = label.replace(" ", "_").lower() + ".pdf"
    plt.savefig(f"figures/{out_pdf}", format="pdf", dpi=150)
    plt.show()


def spline_regression_results():
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
    ctf_estimates = causal_regression(observational_df)
    ctf_end_time = time()
    print(f"CTF time: {ctf_end_time - ctf_start_time}")

    ctf_estimates_df = pd.DataFrame.from_dict(ctf_estimates, orient='index')
    ctf_estimates_df = ctf_estimates_df.rename(columns={0: "estimate"})
    ctf_estimates_df.index.name = "location"
    ctf_estimates_df.to_csv("data/spline_estimates.csv")
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
    print(f"Spline RMSD: {ctf_rmsd}")
    print(f"Spline RMPSE: {ctf_rmspe}")
    print(f"Naive regression RMSD: {naive_regression_rmsd}")
    print(f"Naive regression RMPSE: {naive_regression_rmspe}")

    plot_estimates(gold_standard_ates, naive_ates, ctf_estimates, title="Results using 4680 data points",
                   label="Spline Regression")


if __name__ == "__main__":
    spline_regression_results()
