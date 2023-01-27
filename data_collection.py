import covasim as cv
import covasim.data as cvdata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import argparse
import json
import functools

from covasim.data.loaders import get_age_distribution, get_household_size

VARIANT_BETA_DICT = {'alpha': 1.67*0.016,
                     'beta': 0.016,
                     'gamma': 2.05*0.016,
                     'delta': 2.2*0.016}


BETA_DIST_STANDARD_DEVIATION = 0.016/5


def assign_dominant_variant_and_seed_to_locations(fixed: bool = False, seed: int = 0) -> dict:
    """Assign a dominant COVID variant to each location in Covasim that has a modelled age AND household size.

    :param fixed: Whether to use a pair of fixed variants for all locations or a randomly sampled variant for each
                  location.
    :param seed: Random seed to fix non-deterministic behaviour for reproducibility.
    :return: A dictionary mapping locations to variants from Covasim."""
    random.seed(seed)
    locations_age_and_contacts_data = get_age_and_household_size_for_all_locations()
    locations_seed_and_variant = {}
    variants = list(np.random.choice(list(VARIANT_BETA_DICT.keys()), 2))

    for location in locations_age_and_contacts_data:

        # All locations that have brackets in the name are duplicates e.g. bolivia (plurinational state of) and bolivia
        if not ('(' in location or ')' in location):
            location_label = location.replace(' ', '_') .replace('-', '_').replace('é', 'e')  # For jq compatability
            locations_seed_and_variant[location_label] = {}

            if fixed:
                locations_seed_and_variant[location_label]['variants'] = variants
                seeds = [int(seed) for seed in list(np.random.randint(0, 1e6, 2))]
                locations_seed_and_variant[location_label]['seeds'] = seeds
            else:
                locations_seed_and_variant[location_label]['variant'] = np.random.choice(list(VARIANT_BETA_DICT.keys()))
                locations_seed_and_variant[location_label]['seed'] = np.random.randint(0, 1e6)

    # Remove other duplicates that are synonyms/typos
    del locations_seed_and_variant["viet_nam"]  # "vietnam" exists
    del locations_seed_and_variant["united_states"]  # "usa" exists
    del locations_seed_and_variant["united_states_of_america"]  # "usa" exists

    print(locations_seed_and_variant)
    return locations_seed_and_variant


def variant_to_beta_dist(variant, standard_deviation = BETA_DIST_STANDARD_DEVIATION) -> np.random.normal:
    """Transform variant into a normal distribution with the variant's beta value as mean and 0.016/5 standard dev.

    :param variant: A string representing the COVID-19 variant (alpha, beta, delta, or gamma).
    :param standard_deviation: Standard deviation of the normal distribution from which the beta parameter is drawn.
    :return: A partial function of np.random.normal, preloaded with the parameters for the location's beta dist.
    """
    mean = VARIANT_BETA_DICT[variant]
    return functools.partial(np.random.normal, mean, standard_deviation)


def get_covasim_locations() -> set:
    """Obtain a list of all Covasim locations that have a modelled age distribution OR household size."""
    countries_with_age_distribution = set(get_age_distribution().keys())
    countries_with_household_size = set(get_household_size().keys())
    return countries_with_age_distribution | countries_with_household_size


def age_distribution_to_mean_age(age_distribution: np.array) -> float:
    """Convert a numpy array representation of an age distribution to the mean average age.

    The numpy array has dimensions (n, 3): n rows of age ranges (this differs from country to country) and 3 columns
    specifying the min age, max age, and proportion of population in that age range.

    We calculate a (rough) mean age by taking the mean age of the age range (e.g. for 25 for 20-30 age range) and
    multiplying this value by the proportion (e.g. 25*0.1 for an age range 20-30 that represents 10% of the
    population).
    """
    mean_age_of_age_ranges = np.divide(np.add(age_distribution[:, 0], age_distribution[:, 1]), 2)
    expected_age_of_ranges = np.multiply(mean_age_of_age_ranges, age_distribution[:, 2])
    expected_age = np.sum(expected_age_of_ranges)
    return expected_age


def get_age_and_household_size_for_all_locations(verbose: bool = False) -> dict:
    """Obtain a dictionary mapping all Covasim locations to an age distribution and household size.

    We discard any locations that do not have both an age distribution and a household size
    :param verbose: Whether to print locations missing either an age distribution or household size.
    :return: A dictionary mapping locations with a specified age distribution and household size to those attributes.
    """
    covasim_locations = get_covasim_locations()
    location_age_household_size_dict = {}
    for location in covasim_locations:

        # Get the age distribution of the location
        try:
            age_distribution = get_age_distribution(location)
            mean_age = age_distribution_to_mean_age(age_distribution)
        except ValueError:
            # The location does not have a specified age distribution
            if verbose:
                print(f"Location \"{location}\" does not have a specified age distribution.")
            continue

        # Get the household size of each location
        try:
            household_size = get_household_size(location)
        except ValueError:
            # The location does not have a specified household size
            if verbose:
                print(f"Location \"{location}\" does not have a specified household size.")
            continue

        location_age_household_size_dict[location] = {"age_distribution": age_distribution,
                                                      "mean_age": mean_age,
                                                      "household_size": household_size,
                                                      }
    return location_age_household_size_dict


def plot_location_distributions():
    locations_dict = get_age_and_household_size_for_all_locations()
    fig, ax = plt.subplots(2, 1)

    # Get the mean ages for each location as lists for plotting
    mean_ages = [locations_dict[location]["mean_age"] for location in locations_dict.keys()]

    # Zip locations and ages together and sort by age (ascending)
    locations_and_mean_ages = zip(locations_dict.keys(), mean_ages)
    locations_and_mean_ages_sorted_by_age = sorted(locations_and_mean_ages, key=lambda x: x[1])
    age_sorted_locations, mean_ages_ascending = zip(*locations_and_mean_ages_sorted_by_age)

    # Get the mean household size for each location
    household_sizes = [locations_dict[location]["household_size"] for location in age_sorted_locations]

    # Plot
    ax[0].scatter(age_sorted_locations, mean_ages_ascending, s=.1)
    ax[0].set_ylabel("Mean Age")
    ax[1].scatter(age_sorted_locations, household_sizes, s=.1)
    ax[1].set_ylabel("Mean Household Size")
    ax[0].set_xticks([])
    ax[1].set_xticks([])
    plt.show()


def run_comparison(source_pars_dict: dict, follow_up_pars_dict: dict, outputs_of_interest: [str],
                   n_runs_per_config: int = 1, verbose: int = 1):
    """ Runs Covasim with two different sets of parameters and reports the output of interest for both.

    :param source_pars_dict: The parameter dictionary for the first execution.
    :param follow_up_pars_dict: The parameter dictionary for the second execution.
    :param outputs_of_interest: A list of Covasim outputs that will be recorded in the results.
    :param n_runs_per_config: The number of times to run each configuration.
    :param verbose: Covasim verbose setting (0 for no output, 1 for output).

    :return (source_results_df, follow_up_results_df): A pair of pandas dataframes reporting the outputs of interest for
    the source and follow-up executions, respectively.
    """
    source_results_df = run_sim_with_pars(source_pars_dict, outputs_of_interest, n_runs=n_runs_per_config,
                                          verbose=verbose)
    follow_up_results_df = run_sim_with_pars(follow_up_pars_dict, outputs_of_interest, n_runs=n_runs_per_config,
                                             verbose=verbose)
    return source_results_df, follow_up_results_df


def run_sim_with_pars(pars_dict: dict,
                      desired_outputs: [str],
                      variant: str,
                      n_runs: int = 1,
                      verbose: int = 1,
                      fixed_beta: bool = False,
                      seed: int = 0):
    """ Runs a Covasim COVID-19 simulation with a given dict of parameters and collects the desired outputs, which are
    given as a list of output names.

    :param pars_dict: A dictionary containing the parameters and their values for the run.
    :param desired_outputs: A list of outputs which should be collected.
    :param variant: A string representing the dominant variant of the location to simulate.
    :param n_runs: Number of times to run the simulation with a different seed.
    :param verbose: Covasim verbose setting (0 for no output, 1 for output).
    :param fixed_beta: Whether to use a fixed beta value or produce a distribution centered around the variant's beta.
    :param seed: Random seed for reproducibility. This fixes the stream of random seeds used for each run.
    :return results_df: A pandas df containing the results for each run
    """
    random.seed(seed)
    results_dict = {k: [] for k in list(pars_dict.keys()) + desired_outputs + ['rand_seed', 'avg_age', 'beta',
                                                                               'avg_contacts_h', 'avg_contacts_s',
                                                                               'avg_contacts_w', 'avg_contacts_c',
                                                                               'total_contacts_h', 'total_contacts_s',
                                                                               'total_contacts_w', 'total_contacts_c',
                                                                               'total_contacts', 'agents_in_h',
                                                                               'agents_in_s', 'agents_in_c',
                                                                               'agents_in_w', 'avg_rel_sus']}

    beta_dist = variant_to_beta_dist(variant)
    for _ in range(n_runs):
        # For every run, generate and use a new a random seed. This is to avoid using Covasim's sequential random seeds.
        rand_seed = random.randint(0, 1e6)
        pars_dict['rand_seed'] = rand_seed
        if not fixed_beta:
            pars_dict['beta'] = round(beta_dist(), 5)  # Sample a different beta per repeat from specified distribution
        else:
            pars_dict['beta'] = VARIANT_BETA_DICT[variant]  # A fixed beta value
        sim = cv.Sim(pars=pars_dict, analyzers=[StoreAverageAge(),
                                                StoreContacts(),
                                                StoreAverageRelativeSusceptibility(),
                                                StoreAgentsPerContactLayer()])
        m_sim = cv.MultiSim(sim)
        m_sim.run(n_runs=1, verbose=verbose, n_cpus=1)

        for run in m_sim.sims:
            results = run.results
            # Append inputs to results
            for param in pars_dict.keys():
                results_dict[param].append(run.pars[param])

            # Append outputs to results
            for output in desired_outputs:
                if output not in results:
                    raise IndexError(f'{output} is not in the Covasim outputs.')
                results_dict[output].append(results[output][-1])  # Append the final recorded value for each variable

            # Append average age
            results_dict['avg_age'].append(StoreAverageAge.get_age(run.get_analyzer(label='avg_age')))

            # Append average relative susceptibility
            results_dict['avg_rel_sus'].append(StoreAverageRelativeSusceptibility.get_avg_rel_sus(
                run.get_analyzer(label='avg_rel_sus')))

            # Append household contacts and members
            results_dict['avg_contacts_h'].append(StoreContacts.get_avg_household_contacts(
                run.get_analyzer(label='contacts')))
            results_dict['total_contacts_h'].append(StoreContacts.get_total_household_contacts(
                run.get_analyzer(label='contacts')))
            results_dict['agents_in_h'].append(StoreAgentsPerContactLayer.get_agents_in_household(
                run.get_analyzer(label='agents_per_layer')
            ))

            # Append school contacts and members
            results_dict['avg_contacts_s'].append(StoreContacts.get_avg_school_contacts(
                run.get_analyzer(label='contacts')))
            results_dict['total_contacts_s'].append(StoreContacts.get_total_school_contacts(
                run.get_analyzer(label='contacts')))
            results_dict['agents_in_s'].append(StoreAgentsPerContactLayer.get_agents_in_school(
                run.get_analyzer(label='agents_per_layer')
            ))

            # Append workplace contacts and members
            results_dict['avg_contacts_w'].append(StoreContacts.get_avg_work_contacts(
                run.get_analyzer(label='contacts')))
            results_dict['total_contacts_w'].append(StoreContacts.get_total_work_contacts(
                run.get_analyzer(label='contacts')))
            results_dict['agents_in_w'].append(StoreAgentsPerContactLayer.get_agents_in_workplace(
                run.get_analyzer(label='agents_per_layer')
            ))

            # Append community contacts and members
            results_dict['avg_contacts_c'].append(StoreContacts.get_avg_community_contacts(
                run.get_analyzer(label='contacts')))
            results_dict['total_contacts_c'].append(StoreContacts.get_total_community_contacts(
                run.get_analyzer(label='contacts')))
            results_dict['agents_in_c'].append(StoreAgentsPerContactLayer.get_agents_in_community(
                run.get_analyzer(label='agents_per_layer')
            ))

            # Append total contacts
            results_dict['total_contacts'].append(StoreContacts.get_total_contacts(
                run.get_analyzer(label='contacts')
            ))

    # Any parameters without results are assigned np.nan for each execution
    for param, results in results_dict.items():
        if not results:
            results_dict[param] = [np.nan] * len(results_dict['rand_seed'])
    return pd.DataFrame(results_dict)


def vaccinate_by_age(simulation):
    """A custom method to prioritise vaccination of the elderly. This method is taken from Covasim Tutorial 5:
    https://github.com/InstituteforDiseaseModeling/covasim/blob/7bdf2ddf743f8798fcada28a61a03135d106f2ee/
    examples/t05_vaccine_subtargeting.py

    :param simulation: A covasim simulation for which the elderly will be prioritised for vaccination.
    :return output: A dictionary mapping individuals to vaccine probabilities.
    """
    young = cv.true(simulation.people.age < 50)
    middle = cv.true((simulation.people.age >= 50) * (simulation.people.age < 75))
    old = cv.true(simulation.people.age > 75)
    inds = simulation.people.uid
    vals = np.ones(len(simulation.people))
    vals[young] = 0.1
    vals[middle] = 0.5
    vals[old] = 0.9
    output = dict(inds=inds, vals=vals)
    return output


class StoreAverageAge(cv.Analyzer):
    """ Get the average age of all agents in the simulation on the start day. This is to avoid keeping people from
        simulation runs which requires a lot of memory. """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label = 'avg_age'
        self.avg_age = 0
        return

    def apply(self, sim):
        """ On the first time-step, check the average age of the people in the simulation.

        :param sim: The simulation to which the analyzer is applied.
        """
        if sim.t == 0:
            self.avg_age = np.average(sim.people.age)
        return

    def get_age(self):
        """Return the average age recorded by the analyzer."""
        return round(self.avg_age, 3)


class StoreAgentsPerContactLayer(cv.Analyzer):
    """Get the number of agents in each contact layer on the start day of the simulation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label = 'agents_per_layer'
        self.agents_in_household = 0
        self.agents_in_community = 0
        self.agents_in_school = 0
        self.agents_in_workplace = 0
        return

    def apply(self, sim):
        """ On the first time-step, check the average relative susceptibility of the people in the simulation.

        :param sim: The simulation to which the analyzer is applied.
        """
        if sim.t == 0:
            self.agents_in_household = len(sim.people.contacts['h'].members)
            self.agents_in_community = len(sim.people.contacts['c'].members)
            self.agents_in_school = len(sim.people.contacts['s'].members)
            self.agents_in_workplace = len(sim.people.contacts['w'].members)
        return

    def get_agents_in_household(self):
        """Return the number of agents in the household layer as recorded by the analyzer."""
        return self.agents_in_household

    def get_agents_in_community(self):
        """Return the number of agents in the community layer as recorded by the analyzer."""
        return self.agents_in_community

    def get_agents_in_school(self):
        """Return the number of agents in the school layer as recorded by the analyzer."""
        return self.agents_in_school

    def get_agents_in_workplace(self):
        """Return the number of agents in the workplace layer as recorded by the analyzer."""
        return self.agents_in_workplace


class StoreAverageRelativeSusceptibility(cv.Analyzer):
    """Get the average relative susceptibility of all agents in the simulation on the start day."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label = 'avg_rel_sus'
        self.avg_rel_sus = 0
        return

    def apply(self, sim):
        """ On the first time-step, check the average relative susceptibility of the people in the simulation.

        :param sim: The simulation to which the analyzer is applied.
        """
        if sim.t == sim.pars['n_days']:
            self.avg_rel_sus = np.average(sim.people.rel_sus)
        return

    def get_avg_rel_sus(self):
        """Return the average relative susceptibility recorded by the analyzer."""
        return round(self.avg_rel_sus, 10)


class StoreContacts(cv.Analyzer):
    """ Get the average age of all agents in the simulation on the start day. This is to avoid keeping people from
        simulation runs which requires a lot of memory. """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label = 'contacts'
        self.avg_contacts_h = 0
        self.total_contacts_h = 0
        self.avg_contacts_s = 0
        self.total_contacts_s = 0
        self.avg_contacts_w = 0
        self.total_contacts_w = 0
        self.avg_contacts_c = 0
        self.total_contacts_c = 0
        self.total_contacts = 0
        return

    def apply(self, sim):
        """ On the first time-step, obtain the average contacts of the people in the simulation in each layer.

        :param sim: The simulation to which the analyzer is applied.
        """
        if sim.t == sim.pars['n_days']:
            # Household contacts
            self.avg_contacts_h = round(len(sim.people.contacts['h']['p1']) * 2 /
                                        len(sim.people.contacts['h'].members), 3)
            self.total_contacts_h = round(len(sim.people.contacts['h']['p1']), 3)

            # School contacts
            self.avg_contacts_s = round(len(sim.people.contacts['s']['p1']) * 2 /
                                        len(sim.people.contacts['s'].members), 3)
            self.total_contacts_s = round(len(sim.people.contacts['s']['p1']), 3)

            # Workplace contacts
            self.avg_contacts_w = round(len(sim.people.contacts['w']['p1']) * 2 /
                                        len(sim.people.contacts['w'].members), 3)
            self.total_contacts_w = round(len(sim.people.contacts['w']['p1']), 3)

            # Community contacts
            self.avg_contacts_c = round(len(sim.people.contacts['c']['p1']) * 2 /
                                        len(sim.people.contacts['c'].members), 3)
            self.total_contacts_c = round(len(sim.people.contacts['c']['p1']), 3)

            self.total_contacts = round(len(sim.people.contacts['h']['p1']) +
                                        len(sim.people.contacts['s']['p1']) +
                                        len(sim.people.contacts['w']['p1']) +
                                        len(sim.people.contacts['c']['p1']), 3)
        return

    def get_avg_household_contacts(self):
        """Return the average household contacts recorded by the analyzer."""
        return self.avg_contacts_h

    def get_total_household_contacts(self):
        """Return the total household contacts recorded by the analyzer."""
        return self.total_contacts_h

    def get_avg_school_contacts(self):
        """Return the average school contacts recorded by the analyzer."""
        return self.avg_contacts_s

    def get_total_school_contacts(self):
        """Return the total school contacts recorded by the analyzer."""
        return self.total_contacts_s

    def get_avg_work_contacts(self):
        """Return the average work contacts recorded by the analyzer."""
        return self.avg_contacts_w

    def get_total_work_contacts(self):
        """Return the total work contacts recorded by the analyzer."""
        return self.total_contacts_w

    def get_avg_community_contacts(self):
        """Return the average community contacts recorded by the analyzer."""
        return self.avg_contacts_c

    def get_total_community_contacts(self):
        """Return the total community contacts recorded by the analyzer."""
        return self.total_contacts_c

    def get_total_contacts(self):
        """Return the total contacts across all contact layers recorded by the analyzer."""
        return self.total_contacts


def does_age_affect_infections():
    locations = list(get_age_and_household_size_for_all_locations())
    # some_locations = np.random.choice(locations, 20, replace=False)
    results_dict = {}
    for location in locations:

        sim = cv.Sim(pars={"location": location, "pop_type": "hybrid",
                           "contacts": {"h": 20,
                                        "s": 20,
                                        "w": 20,
                                        "c": 20}
                           },
                     )
        sim.initialize(use_household_data=False)
        sim.run()
        infections = sim.summary['cum_infections']
        age = sim.people.age.mean()
        rel_sus = sim.people.rel_sus.mean()
        a_household_contacts = sim.pars['contacts']['h']
        n_school_contacts = len(sim.people.contacts['s'].members)
        n_workplace_contacts = len(sim.people.contacts['w'].members)

        results_dict[location] = {'cumulative_infections': infections,
                                  'rel_sus': rel_sus,
                                  'n_school_contacts': n_school_contacts,
                                  'n_workplace_contacts': n_workplace_contacts,
                                  'average_age': age,
                                  'a_household_contacts': a_household_contacts}
        if location == "france":
            print(age, a_household_contacts)
    tuples = []
    print(results_dict)
    for k, v in results_dict.items():
        tuples.append((k, v['cumulative_infections'], v['n_school_contacts'],
                       v['a_household_contacts']))
    sorted_tuples = sorted(tuples, key=lambda t: t[2])
    print(sorted_tuples)
    for location_tuple in sorted_tuples:
        plt.scatter(location_tuple[2], location_tuple[1],
                    label=location_tuple[0])
    plt.show()


def do_household_contacts_affect_infections():
    print(get_age_and_household_size_for_all_locations())
    locations = list(get_age_and_household_size_for_all_locations().keys())
    some_locations = np.random.choice(locations, 5, replace=False)
    some_locations = ['UK', 'US-minnesota']
    for location in some_locations:

        results_list = []
        for x in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]:
            sim = cv.Sim(pars={"location": location,
                               "pop_type": "hybrid",
                               "n_days": 200,
                               "rand_seed": 1,
                               "contacts": {"h": x,
                                            "s": 20,
                                            "w": 16,
                                            "c": 20}})
            # sim.initialize(reset=True)
            sim.pars['contacts']['h'] = x
            sim.initialize(reset=True)
            sim.run()
            infections = sim.summary['cum_infections']
            age = sim.people.age.mean()
            contacts = sim.pars['contacts']['h']
            results_list.append((location, infections, contacts))
            # results_dict[location] = {'cumulative_infections': infections,
            #                           'household_contacts': contacts,
            #                           'average_age': age}
        sorted_results = sorted(results_list, key=lambda t: t[2])
        print(sorted_results)
        for location_tuple in sorted_results:
            plt.scatter(location_tuple[2], location_tuple[1],
                        label=location_tuple[0])
        plt.title(location)
        plt.show()


def beta_vs_infections():
    results_list = []
    for beta in [0.01, 0.02, 0.03, 0.04, 0.05]:
        sim = cv.Sim(pars={'location': 'united kingdom',
                           'pop_type': 'hybrid',
                           'beta': beta})
        sim.run()
        infections = sim.summary['cum_infections']
        results_list.append((beta, infections))
    for beta_tuple in results_list:
        plt.scatter(beta_tuple[0], beta_tuple[1])
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    eligible_locations = list(get_age_and_household_size_for_all_locations().keys())
    parser.add_argument('--loc', type=str, default='UK',
                        help=f"Location to simulate. Must be one of the following locations:"
                             f"{eligible_locations}")
    parser.add_argument('--variant', type=str, default='beta',
                        help="The COVID-19 variant to include in the simulation for this location."
                             "This is used to construct a probability distribution from which beta is sampled."
                             "Must be one of the following variants: alpha, beta, delta, gamma.")
    parser.add_argument('--gen', action='store_true',
                        help="Whether to generate a json of beta distributions for each location and a json of beta"
                             " values for each location (for metamorphic testing).")
    parser.add_argument('--seed', type=int, default=0,
                        help="The random seed to use. Must be a positive integer.")
    parser.add_argument('--repeats', type=int, default=10,
                        help="The number of times to run the simulation.")
    parser.add_argument('--fixed', action='store_true',
                        help="Whether to use a fixed variant or a normal distribution centered around its beta value.")
    parser.add_argument('--sd', type=float, default=0.016/5, help="Standard deviation of the beta distributions.")
    args = parser.parse_args()

    if args.gen:
        beta_dists = assign_dominant_variant_and_seed_to_locations(fixed=args.fixed, seed=args.seed)
        if args.fixed:
            json_file_name = f'location_fixed_variants_seed_{args.seed}.json'
        else:
            json_file_name = f'location_variants_seed_{args.seed}.json'
        with open(json_file_name, 'w') as json_file:
            json.dump(beta_dists, json_file, indent=2)
    else:
        location_in_covasim = args.loc.replace('_', ' ')
        if location_in_covasim == 'timor leste':
            location_in_covasim = 'timor-leste'  # Edge case: only valid location with hyphen
        elif location_in_covasim == 'reunion':
            location_in_covasim = 'réunion'

        results_df = run_sim_with_pars({'location': location_in_covasim,
                                        'pop_size': 1e6,
                                        'pop_infected': 1e3,
                                        'start_day': '2020-01-01',
                                        'end_day': '2020-12-31',
                                        'pop_type': 'hybrid'},
                                       desired_outputs=["cum_infections", "cum_deaths"],
                                       variant=args.variant,
                                       n_runs=args.repeats,
                                       fixed_beta=args.fixed,
                                       seed=args.seed)
        # TODO: add variant to output path
        if args.fixed:
            output_path = f'fixed_results/f_{args.loc}_variant_{args.variant}_seed_{args.seed}.csv'
        else:
            output_path = f'results/sd_{args.sd}/{args.loc}_variant_{args.variant}_seed_{args.seed}.csv'
        results_df.to_csv(output_path)
        print(f'Saving results to {output_path}...')

