import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from ctf_application import increasing_beta

figure(figsize=(25, 8), dpi=100)


def gold_standard_results(df):
    gold_standard_dict = {}
    for location in df.index:
        gold_standard_dict[location] = df.at[location, "change_in_infections"]
    sorted_gold_standard_dict = {k: v for k, v in
                                 sorted(gold_standard_dict.items())}
    return sorted_gold_standard_dict


def naive_regression(df):
    naive_regression_eqn = "cum_infections ~ np.log(beta)"
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
    # causal_regression_eqn = """cum_infections ~ beta + np.power(beta, 2) +
    #                            avg_contacts_s + avg_contacts_w + avg_rel_sus +
    #                            avg_contacts_h + beta:avg_contacts_s +
    #                            beta:avg_contacts_w + beta:avg_contacts_h +
    #                            beta:avg_rel_sus
    #                        """

    # PROCESS TO CONSTRUCT THIS REGRESSION EQUATION:
    # First, we used our causal dag to identify sufficient adjustment sets.
    # We then used our domain knowledge to select, from these adjustment sets,
    # which is most appropriate. In this case, using total contacts in workplace
    # and school, and avg rel sus is better than age because it's the shape of
    # the age distribution that matters, not the mean.
    # We also add quadratic terms where variables are known or suspected to have
    # a curvilinear relationship with cumulative infections. This may not be
    # strictly necessary after linearising the data, but is there for safety.

    # causal_regression_eqn = """cum_infections ~ np.log(beta) + np.power(np.log(beta), 2) +
    #                                np.log(avg_contacts_h) + np.power(np.log(avg_contacts_h), 2) +
    #                                np.log(avg_rel_sus) + np.power(np.log(avg_rel_sus), 2) +
    #                                np.log(total_contacts_w) + np.power(np.log(total_contacts_w), 2) +
    #                                np.log(total_contacts_s) + np.power(np.log(total_contacts_s), 2) +
    #                                np.log(beta):np.log(total_contacts_w) +
    #                                np.log(beta):np.log(total_contacts_s) +
    #                                np.log(beta):np.log(avg_contacts_h) +
    #                                np.log(beta):np.log(avg_rel_sus)
    #                            """
    to_log = ["beta", "avg_contacts_h", "avg_rel_sus", "total_contacts_w",
              "total_contacts_s"]
    logged_df = df.copy()
    logged_df[to_log] = np.log(logged_df[to_log])
    print(logged_df)
    print(df)
    causal_regression_eqn = """cum_infections ~ beta + np.power(beta, 2) +
                                       avg_contacts_h + np.power(avg_contacts_h, 2) +
                                       avg_rel_sus + np.power(avg_rel_sus, 2) +
                                       total_contacts_w + np.power(total_contacts_w, 2) + 
                                       total_contacts_s + np.power(total_contacts_s, 2) +
                                       beta:total_contacts_w +
                                       beta:total_contacts_s +
                                       beta:avg_contacts_h +
                                       beta:avg_rel_sus
                                   """
    causal_model = smf.ols(causal_regression_eqn, data=logged_df).fit()
    source_follow_up_dict = {'beta': [np.log(0.016), np.log(0.02672)]}
    # adjustment_set = ["avg_contacts_h", "avg_contacts_w", "avg_contacts_s",
    #                   "avg_rel_sus"]
    adjustment_set = ["avg_contacts_h", "avg_age", "avg_rel_sus", "total_contacts_s",
                      "total_contacts_w"]
    locations = df["location"].unique()
    results_dict = {}
    for location in locations:
        adjustment_dict = {}
        location_df = logged_df.loc[logged_df["location"] == location]
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
    difference = np.array(true_values) - np.array(estimates)
    squared_difference = np.power(difference, 2)
    sum_squared_difference = np.sum(squared_difference)
    normalised_sum_squared_difference = sum_squared_difference/len(estimates)
    rmsd = np.sqrt(normalised_sum_squared_difference)
    return rmsd


def rmsd_from_dicts(true_dict, estimate_dict):
    true_values = []
    estimates = []
    for location in true_dict.keys():
        true_values.append(true_dict[location])
        estimates.append(estimate_dict[location])
    return rmsd(true_values, estimates)


def individual_errors(true_dict, estimate_dict):
    errors = {}
    for location in true_dict.keys():
        errors[location] = estimate_dict[location] - true_dict[location]
    return errors


def plot_estimates(gold_standard, naive_estimates, causal_estimates):
    ascending_gold_standard = {k: v for k, v in sorted(gold_standard.items(),
                                                       key=lambda item: item[1]
                                                       )}

    xs = list(ascending_gold_standard.keys())
    ys_true = [ascending_gold_standard[location] for location in xs]
    ys_naive = [naive_estimates[location] for location in xs]
    ys_causal = [causal_estimates[location] for location in xs]

    plt.scatter(xs, ys_true, label="Gold Standard", marker='.')
    plt.scatter(xs, ys_naive, label="Standard Regression", marker='.')
    plt.scatter(xs, ys_causal, label="Causal Regression", marker='.')
    plt.xticks(rotation=60, ha='right')
    plt.ylabel("Change in Cumulative Infections")
    plt.title("Predicting Metamorphic Test Outcomes From Observational Data: 20% of Original Data.")
    plt.plot()
    plt.legend()
    plt.tight_layout()
    plt.savefig("ctf_results.pdf", format="pdf", dpi=300)
    plt.show()


if __name__ == "__main__":
    # Regression equations
    observational_df = pd.read_csv("results/varied_sds/sd_0.002.csv",
                                   index_col=0)
    gold_standard_df = pd.read_csv("smt_results.csv", index_col=0)
    naive_ates = naive_regression(observational_df)
    causal_ates = causal_regression(observational_df)
    gold_standard_ates = gold_standard_results(gold_standard_df)
    # print("Naive ATEs: ")
    # print(naive_ates)
    # print("Naive errors: ")
    # naive_errors = individual_errors(gold_standard_ates, naive_ates)
    # print(naive_errors)
    # print("Naive RMSD: ")
    # print(rmsd_from_dicts(gold_standard_ates, naive_ates))
    # print("Causal ATEs: ")
    # print(causal_ates)
    # print("Causal errors: ")
    # causal_errors = individual_errors(gold_standard_ates, causal_ates)
    # print(causal_errors)
    # print("Causal RMSD: ")
    # print(rmsd_from_dicts(gold_standard_ates, causal_ates))
    # print(causal_ates)

    ctf_estimates = increasing_beta("results/different_sized_data/overall/sd_0.002/size_0.9.csv")
    plot_estimates(gold_standard_ates, naive_ates, ctf_estimates)
    print(rmsd_from_dicts(gold_standard_ates, ctf_estimates))



