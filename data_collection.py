import covasim as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random


from covasim.data.loaders import get_age_distribution, get_household_size


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


def get_age_and_household_size_for_all_locations() -> dict:
    """Obtain a dictionary mapping all Covasim locations to an age distribution and household size.

    We discard any locations that do not have both an age distribution and a household size."""
    covasim_locations = get_covasim_locations()
    location_age_household_size_dict = {}
    for location in covasim_locations:

        # Get the age distribution of the location
        try:
            age_distribution = get_age_distribution(location)
            mean_age = age_distribution_to_mean_age(age_distribution)
        except ValueError:
            # The location does not have a specified age distribution
            print(f"Location \"{location}\" does not have a specified age distribution.")
            continue

        # Get the household size of each location
        try:
            household_size = get_household_size(location)
        except ValueError:
            # The location does not have a specified household size
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


def collect_observational_data(n_runs_per_location: int = 10):
    """Run Covasim to obtain simulated observational data.

    For each location with an available age distribution and household contact data, we run the model n times.

    :return:
    """
    age_and_household_data = get_age_and_household_size_for_all_locations()
    print(len(age_and_household_data.keys()))
    print(n_runs_per_location)
    pass


def run_comparison(source_pars_dict: dict, follow_up_pars_dict: dict, outputs_of_interest: [str],
                   n_runs_per_config: int = 1, verbose: int = -1):
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


def run_sim_with_pars(pars_dict: dict, desired_outputs: [str], n_runs: int = 1, verbose: int = -1,
                      seed: int = 0):
    """ Runs a Covasim COVID-19 simulation with a given dict of parameters and collects the desired outputs, which are
    given as a list of output names.

    :param pars_dict: A dictionary containing the parameters and their values for the run.
    :param desired_outputs: A list of outputs which should be collected.
    :param n_runs: Number of times to run the simulation with a different seed.
    :param verbose: Covasim verbose setting (0 for no output, 1 for output).
    :param seed: Random seed for reproducibility. This fixes the stream of random seeds used for each run.
    :return results_df: A pandas df containing the results for each run
    """
    random.seed(seed)
    results_dict = {k: [] for k in list(pars_dict.keys()) + desired_outputs + ['rand_seed', 'avg_age', 'beta',
                                                                               'avg_contacts_h', 'avg_contacts_s',
                                                                               'avg_contacts_w', 'avg_contacts_c']}
    for _ in range(n_runs):
        # For every run, generate and use a new a random seed. This is to avoid using Covasim's sequential random seeds.
        rand_seed = random.randint(0, 1e6)
        pars_dict['rand_seed'] = rand_seed
        sim = cv.Sim(pars=pars_dict, analyzers=[StoreAverageAge(), StoreContacts()])
        m_sim = cv.MultiSim(sim)
        m_sim.run(n_runs=1, verbose=verbose, n_cpus=1)

        for run in m_sim.sims:
            results = run.results
            # Append inputs to results
            for param in pars_dict.keys():
                # if param == 'variants':
                #     variant = run.pars[param][0].label
                #     results_dict[param].append(variant)
                #     # Beta is obtained by multiplying the base beta value by the variant-specific beta factor
                #     results_dict['wild_beta'].append(run.pars['beta'])
                #     results_dict['rel_beta'].append(run.pars['variant_pars'][variant]['rel_beta'])
                #     results_dict['beta'].append(run.pars['variant_pars'][variant]['rel_beta']*run.pars['beta'])
                #     results_dict['rel_severe_prob'].append(run.pars['variant_pars'][variant]['rel_severe_prob'])
                # else:
                results_dict[param].append(run.pars[param])

            # # If a hybrid population is used, also record the household contacts
            # if pars_dict['pop_type'] == 'hybrid':
            #     results_dict['contacts_h'] = run.pars['contacts']['h']
            #     results_dict['contacts_s'] = run.pars['contacts']['s']
            #     results_dict['contacts_w'] = run.pars['contacts']['w']
            #     results_dict['contacts_c'] = run.pars['contacts']['c']
            # else:
            #     results_dict['contacts'] = run.pars['contacts']['a']

            # If variant has been specified, change pop_infected to n_imports
            # if 'variants' in pars_dict:
            #     results_dict['pop_infected'][-1] = run.pars['variants'][0].n_imports

            # Append outputs to results
            for output in desired_outputs:
                if output not in results:
                    raise IndexError(f'{output} is not in the Covasim outputs.')
                results_dict[output].append(results[output][-1])  # Append the final recorded value for each variable
            # Append average age
            results_dict['avg_age'].append(StoreAverageAge.get_age(run.get_analyzer(label='avg_age')))

            # Append average contacts
            results_dict['avg_contacts_h'].append(StoreContacts.get_avg_household_contacts(
                run.get_analyzer(label='avg_contacts')))
            results_dict['avg_contacts_s'].append(StoreContacts.get_avg_school_contacts(
                run.get_analyzer(label='avg_contacts')))
            results_dict['avg_contacts_w'].append(StoreContacts.get_avg_work_contacts(
                run.get_analyzer(label='avg_contacts')))
            results_dict['avg_contacts_c'].append(StoreContacts.get_avg_community_contacts(
                run.get_analyzer(label='avg_contacts')))

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


class StoreContacts(cv.Analyzer):
    """ Get the average age of all agents in the simulation on the start day. This is to avoid keeping people from
        simulation runs which requires a lot of memory. """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label = 'avg_contacts'
        self.contacts_h = 0
        self.contacts_s = 0
        self.contacts_w = 0
        self.contacts_c = 0
        return

    def apply(self, sim):
        """ On the first time-step, obtain the average contacts of the people in the simulation in each layer.

        :param sim: The simulation to which the analyzer is applied.
        """
        if sim.t == sim.pars['n_days']:
            self.contacts_h = round(len(sim.people.contacts['h']['p1']) * 2 /
                                    len(sim.people.contacts['h'].members), 3)
            self.contacts_s = round(len(sim.people.contacts['s']['p1']) * 2 /
                                    len(sim.people.contacts['s'].members), 3)
            self.contacts_w = round(len(sim.people.contacts['w']['p1']) * 2 /
                                    len(sim.people.contacts['w'].members), 3)
            self.contacts_c = round(len(sim.people.contacts['c']['p1']) * 2 /
                                    len(sim.people.contacts['c'].members), 3)
        return

    def get_avg_household_contacts(self):
        """Return the average household contacts recorded by the analyzer."""
        return self.contacts_h

    def get_avg_school_contacts(self):
        """Return the average school contacts recorded by the analyzer."""
        return self.contacts_s

    def get_avg_work_contacts(self):
        """Return the average work contacts recorded by the analyzer."""
        return self.contacts_w

    def get_avg_community_contacts(self):
        """Return the average community contacts recorded by the analyzer."""
        return self.contacts_c


if __name__ == "__main__":

    print(collect_observational_data())
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     results_df = run_sim_with_pars({'location': 'UK', 'pop_size': 1e5, 'pop_infected': 1e3,
    #                                     'start_day': '2020-01-01', 'end_day': '2020-04-01',
    #                                     'beta': 0.05, 'pop_type': 'hybrid'},
    #                                    desired_outputs=["cum_infections", "cum_deaths"],
    #                                    n_runs=3,
    #                                    seed=1)
    #     print(results_df)
