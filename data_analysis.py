"""A script to analyse the collected data."""
import pandas as pd
import glob
import os
from causal_testing.specification.causal_specification import CausalSpecification, CausalDAG, Scenario
from causal_testing.specification.scenario import Input, Output
from causal_testing.testing.causal_test_engine import CausalTestEngine, CausalTestCase
from causal_testing.testing.estimators import LinearRegressionEstimator
from causal_testing.testing.causal_test_outcome import Positive
from causal_testing.data_collection.data_collector import ObservationalDataCollector


def combine_results(results_dir: str):
    """Combine results into a single dataframe.

    :param results_dir: Path to the directory containing results CSVs.
    :return: A dataframe containing all data in the CSVs.
    """
    all_files = glob.glob(os.path.join(results_dir, "*.csv"))
    print(all_files)
    dfs = [pd.read_csv(csv_file, index_col=0) for csv_file in all_files]
    print(dfs)
    return pd.concat(dfs, ignore_index=True)


def doubling_beta(control_beta: float, treatment_beta: float):

    # 1. Create the causal specification
    # DAG
    dag = CausalDAG("dag.dot")

    # Varying inputs
    location = Input('location', str)
    beta = Input('beta', float)

    # Fixed inputs (hence not in DAG)
    pop_size = Input('pop_size', float)
    pop_type = Input('pop_type', str)
    pop_infected = Input('pop_infected', float)
    start_day = Input('start_day', str)
    end_day = Input('end_day', str)

    # Outputs
    avg_contacts_h = Output('avg_contacts_h', float)
    avg_age = Output('avg_age', float)
    total_contacts_h = Output('total_contacts_h', int)
    avg_rel_sus = Output('avg_rel_sus', float)
    total_contacts_w = Output('total_contacts_w', int)
    total_contacts_s = Output('total_contacts_s', int)
    total_contacts_c = Output('total_contacts_c', int)
    cum_infections = Output('cum_infections', int)

    # Scenario
    scenario = Scenario(variables=[location, beta, pop_size, pop_type,
                                   pop_infected, start_day, end_day,
                                   avg_contacts_h, avg_age, total_contacts_h,
                                   avg_rel_sus, total_contacts_w,
                                   total_contacts_s, total_contacts_c,
                                   cum_infections],
                        constraints=[pop_size.z3 == 1e6, pop_infected.z3 == 1e3,
                                     start_day.z3 == '2020-01-01',
                                     end_day.z3 == '2020-12-31',
                                     pop_type.z3 == 'hybrid'])

    # Specification
    causal_specification = CausalSpecification(scenario, dag)

    causal_test_case = CausalTestCase(
        control_input_configuration={beta: control_beta},
        expected_causal_effect=Positive,
        treatment_input_configuration={beta: treatment_beta},
        outcome_variables={cum_infections})

    # 6. Create a data collector
    data_collector = ObservationalDataCollector(scenario,
                                                "results/complete_data/"
                                                "passively_observed.csv")
    df = data_collector.collect_data()

    # 7. Create an instance of the causal test engine
    causal_test_engine = CausalTestEngine(causal_specification, data_collector)

    # 8. Obtain the minimal adjustment set for the causal test case from the causal DAG
    causal_test_engine.identification(causal_test_case)

    linear_regression_estimator = LinearRegressionEstimator(('beta',), 0.032, 0.016,
                                                            {'total_contacts_w', 'total_contacts_s',
                                                             'total_contacts_h', 'total_contacts_c',
                                                             'avg_rel_sus'},
                                                            ('cum_infections',),
                                                            df=df)

    # Add squared terms for beta, since it has a quadratic relationship with cumulative infections
    linear_regression_estimator.add_squared_term_to_df('beta')
    causal_test_result = causal_test_engine.execute_test(linear_regression_estimator, causal_test_case, 'ate')
    print(causal_test_result)


if __name__ == "__main__":
   # doubling_beta(0.016, 0.032)
    df = combine_results("fixed_results")
    df.to_csv("fixed_results/complete_data/smt.csv")
