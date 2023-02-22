import numpy as np
import random
import pandas as pd
from causal_testing.testing.causal_test_case import CausalTestCase, BaseTestCase
from causal_testing.specification.causal_dag import CausalDAG, Scenario
from causal_testing.specification.causal_specification import CausalSpecification
from causal_testing.specification.variable import Input, Output
from causal_testing.data_collection.data_collector import ObservationalDataCollector
from causal_testing.testing.causal_test_outcome import Positive
from causal_testing.testing.causal_test_engine import CausalTestEngine
from causal_testing.testing.estimators import LinearRegressionEstimator


def less_data_per_location(csv_path: str, proportion: float):
    random.seed(0)
    df = pd.read_csv(csv_path)
    locations = df["location"].unique()
    for location in locations:
        location_df = df.loc[df["location"] == location]
        location_df_to_remove = location_df.sample(frac=1-proportion)
        df = pd.concat([df, location_df_to_remove]).drop_duplicates(keep=False)
    return df


def less_data_overall(csv_path: str, proportion: float):
    random.seed(0)
    df = pd.read_csv(csv_path)
    return df.sample(frac=proportion)


def increasing_beta(csv_path: str):
    # We start by creating our Causal Specification. This means we need to provide the causal DAG, the inputs, the
    # outputs, and any constraints.
    dag = CausalDAG("dag.dot")
    ref_df = pd.read_csv("results/varied_sds/sd_0.002.csv")

    location = Input('location', str)
    beta = Input('beta', float)
    pop_size = Input('pop_size', float)
    pop_type = Input('pop_type', str)
    pop_infected = Input('pop_infected', float)
    start_day = Input('start_day', str)
    end_day = Input('end_day', str)
    avg_contacts_h = Output('avg_contacts_h', float)
    avg_rel_sus = Output('avg_rel_sus', float)
    total_contacts_w = Output('total_contacts_w', int)
    total_contacts_s = Output('total_contacts_s', int)
    cum_infections = Output('cum_infections', int)

    scenario = Scenario(variables=[location, beta, pop_size, pop_type, pop_infected, start_day, end_day, avg_contacts_h,
                                   avg_rel_sus, total_contacts_w, total_contacts_s, cum_infections],
                        constraints=[pop_size.z3 == 1e6, pop_infected.z3 == 1e3,
                                     start_day.z3 == '2020-01-01',
                                     end_day.z3 == '2020-12-31',
                                     pop_type.z3 == 'hybrid'])

    causal_specification = CausalSpecification(scenario, dag)

    # Next we specify our Causal Test Case. We start by selecting the treatment variable (the one we intervene on, which
    # is beta in our case) and the outcome variable (the one that we expect will be affected by the intervention, which
    # is cumulative infections in our case).
    btc = BaseTestCase(treatment_variable=beta,
                       outcome_variable=cum_infections)

    # We complete our test case with the control and treatment value, as well as the expected effect (positive change).
    # The control and treatment value define the intervention. We apply a log transformation to our control and
    # treatment value here because we log transform our independent variables to linearise the data.
    causal_test_case = CausalTestCase(
        base_test_case=btc,
        expected_causal_effect=Positive,
        control_value=np.log(0.016),
        treatment_value=np.log(0.02672)
    )

    # We then provide our observational data as an existing CSV.
    data_collector = ObservationalDataCollector(scenario, csv_path)
    df = data_collector.collect_data()  # TODO: Should this not be called automatically upon instantiation?

    # We then create an instance of the causal test engine, which brings together the causal knowledge of the causal
    # specification, the data, and the causal test case.
    causal_test_engine = CausalTestEngine(causal_specification, data_collector)

    # Next we specify our adjustment set. We could use the minimal adjustment set or specify our own custom one.
    minimal_adjustment_set = dag.identification(btc)  # We use our own.
    print(minimal_adjustment_set)
    adjustment_set = {'avg_contacts_h', 'total_contacts_w', 'total_contacts_s', 'avg_rel_sus'}

    # Log transform independent variables to linearise the data
    variables_to_log = list(adjustment_set)
    variables_to_log.append('beta')
    df[variables_to_log] = np.log(df[variables_to_log])
    ref_df[variables_to_log] = np.log(ref_df[variables_to_log])

    # Specify product terms (i.e. interactions)
    product_terms = [('beta', av) for av in adjustment_set]
    effect_mod_vars = [avg_contacts_h, total_contacts_w, total_contacts_s, avg_rel_sus]
    locations = ref_df["location"].unique()

    # Perform a separate causal test case for each location in the data. We know that the correct metamorphic test
    # outcome will vary from location to location. Therefore, we want to estimate the effect for each location
    # separately.
    ctf_estimates = {}
    for location in locations:
        effect_modifiers = {av: ref_df.loc[ref_df["location"] == location][av.name].mean() for av in effect_mod_vars}
        linear_regression_estimator = LinearRegressionEstimator(('beta',),
                                                                control_value=np.log(0.016),  # This is re-specified
                                                                treatment_value=np.log(0.02672),  # This is re-specified
                                                                adjustment_set=adjustment_set,
                                                                outcome=('cum_infections',),
                                                                product_terms=product_terms,
                                                                effect_modifiers=effect_modifiers,
                                                                df=df)

        # Add squared terms for all independent variables to capture suspected curvilinear relations.
        for av in ['beta', 'avg_contacts_h', 'total_contacts_w', 'total_contacts_s', 'avg_rel_sus']:
            linear_regression_estimator.add_squared_term_to_df(av)
        causal_test_result = causal_test_engine.execute_test(linear_regression_estimator,
                                                             causal_test_case,
                                                             'ate_calculated')
        print(causal_test_result)
        ctf_estimates[location] = causal_test_result.test_value.value
    return ctf_estimates


if __name__ == "__main__":
    estimates = increasing_beta("data/observational_data.csv")
