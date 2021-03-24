import scipy
import pandas as pd
from scipy import integrate
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import epintervene.simobjects as epi
import time
import diseasemodel.model as model
from coviddata import dataexplore
# TODO next: Inferring the right transmission probabilities for the simulation from the rate parameters

t_res = []
time_series = []
beta = 0.625  # An infected person infects a person every 2 days (half a person per day)
sigma = 0.5
gamma = 1 / 10  # Recover period is 5 days
gamma_ei = 1 / 6.7  # latent period is 5 days, say
N=100
def playing_around():
    # Toy SEIR model to get things going
    y_sir = np.array([99/100, 1/100, 0])
    y_seir = np.array([98/100, 1/100, 0/100, 0])
    t=0
    # my_solver = scipy.integrate.RK45(my_odes_sir, t0=t, y0=y_sir, t_bound=100)
    # for i in range(12):
    #     my_solver.step()
    # plt.show()

    solution = scipy.integrate.solve_ivp(my_odes_seir, t_span=[0, 100000], y0=y_seir)
    # t_res = solution.t
    time_series.append(solution.t)
    time_series.append(solution.y[0])
    time_series.append(solution.y[1])
    time_series.append(solution.y[2])
    time_series.append(solution.y[3]) # If Exposed

    return solution

def my_odes_sir(t, y):
    # y form: [s(t), i(t), r(t)]
    s_t = y[0]
    i_t = y[1]
    r_t = y[2]
    ds_dt = - beta*(s_t*i_t)
    di_dt = beta*(s_t*i_t) - gamma*i_t
    dr_dt = gamma*i_t
    print([s_t+ds_dt, i_t+di_dt, r_t+dr_dt])
    new_y = np.array([ds_dt, di_dt, dr_dt])
    return new_y

def my_odes_seir(t, y):
    # y form: [s(t), e(t), i(t), r(t)]
    s_t = y[0]
    e_t = y[1]
    i_t = y[2]
    r_t = y[3]
    ds_dt = -sigma*beta*(s_t*e_t) - beta*(s_t*i_t)
    de_dt = sigma*beta*(s_t*e_t) + beta*(s_t*i_t) - gamma_ei*e_t
    di_dt = -gamma*i_t + gamma_ei*e_t
    dr_dt = gamma*i_t
    print(t)
    # t_res.append(t)
    # time_series.append([t, s_t, e_t, i_t, r_t])
    print([s_t+ds_dt, e_t+de_dt, i_t+di_dt, r_t+dr_dt])
    print(sum([s_t+ds_dt, e_t+de_dt, i_t+di_dt, r_t+dr_dt]))
    # plt.scatter(t, s_t, color='blue')
    # plt.scatter(t, e_t, color='orange')
    # plt.scatter(t, i_t, color='red')
    # plt.scatter(t, r_t, color='green')
    new_y = np.array([ds_dt, de_dt, di_dt, dr_dt])
    return new_y

def sample_sim(A):
    little_b = -(gamma * beta) / (beta - 1)
    Beta = np.full((N, N), beta/N)
    np.fill_diagonal(Beta, 0)
    # Weirdly, what matches the ODE model well is 1.25*beta/N for probability of transmission, with contacts of network=1
    Gamma = np.full((N), gamma)
    sir_sim = epi.Simulation(A)
    sir_sim.add_infection_event_rates(Beta)
    sir_sim.add_recover_event_rates(Gamma)
    sir_sim.run_sim(wait_for_recovery=True)
    return sir_sim.tabulate_continuous_time(custom_range=True, custom_t_lim=45)

def sample_sim_seir(A, adj_list):
    Beta = np.full((N, N), 1.0/N)
    # Beta = np.full((N, N), beta/N)
    np.fill_diagonal(Beta, 0)
    Beta_SE = np.full((N, N), 1.0*sigma/N)
    # Beta_SE = np.full((N, N), sigma*beta/N)
    np.fill_diagonal(Beta_SE, 0)
    Gamma_EI = np.full((N), gamma_ei) #TODO does this end up sounding right for the model, if you make gamma 1 for ex. then it ends up as 1 per day
    Gamma = np.full((N), gamma)

    seir_sim = epi.SimulationSEIR(N=N, adjmatrix=A, adjlist=adj_list)
    # seir_sim = epi.Simulation(A)
    seir_sim.add_infection_event_rates(Beta)
    seir_sim.add_exposed_event_rates(Beta_SE)
    seir_sim.add_exposed_infected_event_rates(Gamma_EI)
    seir_sim.add_recover_event_rates(Gamma)
    seir_sim.run_sim(wait_for_recovery=True)
    return seir_sim.tabulate_continuous_time(custom_range=True, custom_t_lim=45)

def seir_sim_for_detention_staff_county(N=1000, A=None, adj_list=None):

    # Make the membership groups
    node_mems = []
    for i in range(50):
        node_mems.append('C')
    for j in range(25):
        node_mems.append('D')
    for k in range(25):
        node_mems.append('O')

    memb_groups = ['C', 'D', 'O']
    # TODO need to define the SBM network for this

    seir_sim = epi.SimulationSEIR(N=N, adjmatrix=A, adjlist=adj_list, membership_groups=memb_groups, node_memberships=node_mems)
    # seir_sim = epi.Simulation(A)
    # seir_sim.add_infection_event_rates(Beta)
    # seir_sim.add_exposed_event_rates(Beta_SE)
    # seir_sim.add_exposed_infected_event_rates(Gamma_EI)
    # seir_sim.add_recover_event_rates(Gamma)
    seir_sim.set_uniform_beta(1/N)
    seir_sim.set_uniform_beta_es(sigma/N) #todo will this work?
    seir_sim.set_uniform_gamma_ei(gamma_ei)
    seir_sim.set_uniform_gamma(gamma)

    print('Starting sim')
    start_time = time.time()
    seir_sim.run_sim(wait_for_recovery=True, uniform_rate=True, with_memberships=True)
    print(f'Single sim time {time.time()-start_time}')
    return seir_sim.tabulate_continuous_time_with_groups(custom_range=True, custom_t_lim=45)
    # return seir_sim.tabulate_continuous_time(custom_range=True, custom_t_lim=45)

def plot_ensemble_with_memb_groups(num_sims=50, show=False):
    plt.figure(1)
    net = nx.erdos_renyi_graph(N, p=beta)
    # A = np.array(nx.adjacency_matrix(net).todense())
    # A = np.full((N, N), 1)
    # np.fill_diagonal(A, 0)
    adj_list = epi.network.NetworkBuilder.create_adjacency_list(net)
    ts, infected_dict, exposed_dict = seir_sim_for_detention_staff_county(N, None, adj_list=adj_list)
    # time_partition, infection_time_series, recover_time_series = sample_sim(A)
    ensemble_res = np.zeros((2, len(ts)))

    sims_counted = 0
    for i in range(num_sims):
        if i % 50 == 0:
            net = nx.erdos_renyi_graph(N, p=beta)
            # A = np.array(nx.adjacency_matrix(net).todense())
            adj_list = epi.network.NetworkBuilder.create_adjacency_list(net)
            # A = np.full((N, N), 1)
            # np.fill_diagonal(A, 0)
        ts, inf_result, exp_result = seir_sim_for_detention_staff_county(N, None, adj_list)
        # tim, inft, rec = sample_sim(A)
        # only count the epidemics that "take off"
        if inf_result['C'][20]!=1:
            exposed_dict['C'] += exp_result['C']
            exposed_dict['D'] += exp_result['D']
            exposed_dict['O'] += exp_result['O']
            infected_dict['C'] += inf_result['C']
            infected_dict['D'] += inf_result['D']
            infected_dict['O'] += inf_result['O']
            sims_counted += 1
        # ensemble_res[0] += inft
        # ensemble_res[1] += rec
        # ensemble_res[2] += exp_ts
        # sims_counted += 1
        # ensemble_res[2] += exp_ts
    exposed_dict['C'] = exposed_dict['C']/sims_counted
    exposed_dict['D'] = exposed_dict['D']/sims_counted
    exposed_dict['O'] = exposed_dict['O']/sims_counted
    infected_dict['C'] = infected_dict['C']/sims_counted
    infected_dict['D'] = infected_dict['D']/sims_counted
    infected_dict['O'] = infected_dict['O']/sims_counted

    # plt.plot(time_partition, time_series[1], color='blue', label='Susceptible')
    # plt.plot(time_partition, exposed_time_series/N, color='orange', label='Exposed')
    plt.plot(ts, exposed_dict['C']/N, color='orange', label='C Exposed', ls=':')
    plt.plot(ts, exposed_dict['D']/N, color='orange', label='D Exposed', ls='-')
    plt.plot(ts, exposed_dict['O']/N, color='orange', label='O Exposed', ls='-.')

    plt.plot(ts, infected_dict['C']/N, color='red', label='C Infected', ls=':')
    plt.plot(ts, infected_dict['D']/N, color='red', label='D Infected', ls='-')
    plt.plot(ts, infected_dict['O']/N, color='red', label='O Infected', ls='-.')
    # plt.plot(time_partition, ensemble_res[2]/N, color='orange', label='Exposed')
    # plt.ylim([0,1])
    plt.legend(loc='upper left')
    plt.tight_layout()
    if show:
        plt.show()



if __name__ == '__main__':
    ### Example:
    # Plot model for Adelanto facility, and san bernardino county
    plt.figure('model')
    model.example_model(show_recovered=True) # Something isn't quite right here in terms of the spreading rates, for large county/small subpopulation
    plt.show()
    # Plot covid cases from NYT for SB county and from scraped data for Adelanto, in September
    # See if numbers match up, also with data from the article I found
    ucla_data = dataexplore.select_ice_facilities(dataexplore.load_ucla_data())
    adelanto = ucla_data[ucla_data['Name'].str.contains('ADELANTO')]
    adelanto['Date'] = pd.to_datetime(adelanto['Date'])
    covid_data = dataexplore.load_nyt_data()
    san_bern = covid_data[covid_data['county'].str.contains('San Bernardino')]
    san_bern['date'] = pd.to_datetime(san_bern['date'])

    start_date = '2020-09-01'
    end_date = '2020-09-30'
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    covid_data_mask = (san_bern['date'] > start_date) & (san_bern['date'] <= end_date)
    san_bern = san_bern.loc[covid_data_mask]
    covid_data_mask_2 = (adelanto['Date'] > start_date) & (adelanto['Date'] <= end_date)
    adelanto = adelanto.loc[covid_data_mask_2]
    adelanto.dropna()

    plt.figure('adelanto')
    plt.plot(adelanto['Date'], adelanto['Residents.Confirmed'], color='blue')
    # plt.xlabel(adelanto['Date'], rotation=45)
    plt.figure('county')
    plt.plot(san_bern['date'], san_bern['cases'], color='orange')
    plt.show()

    ### End Example
    ucla_data = dataexplore.select_ice_facilities(dataexplore.load_ucla_data())
    plot_ensemble_with_memb_groups(num_sims=300)
    model.solve_and_plot(N=100)

    playing_around()
    plt.plot(t_res)
    plt.scatter(np.arange(len(t_res)),t_res)
    plt.show()
    t_res = np.array(t_res)
    plt.plot(t_res[1:]-t_res[:len(t_res)-1])
    plt.show()
    time_series = np.array(time_series)
    # time_series = time_series.T
    plt.figure(0)
    plt.plot(time_series[0][:15], time_series[1][:15], color='blue', label='Susceptible')
    plt.plot(time_series[0][:15], time_series[2][:15], color='orange', label='Exposed')
    plt.plot(time_series[0][:15], time_series[3][:15], color='red', label='Infected')
    plt.plot(time_series[0][:15], time_series[4][:15], color='green', label='Recovered')
    # plt.xticks(time_series[0], rotation=45, fontsize=10)
    plt.legend(loc='upper left')
    plt.ylim([0, 1])
    plt.tight_layout()

    # plt.show()

    plt.figure(1)
    net = nx.erdos_renyi_graph(N, p=beta)
    A = np.array(nx.adjacency_matrix(net).todense())
    # A = np.full((N, N), 1)
    # np.fill_diagonal(A, 0)
    adj_list = epi.network.NetworkBuilder.create_adjacency_list(net)
    time_partition, infection_time_series, recover_time_series, exposed_time_series = sample_sim_seir(A, adj_list=adj_list)
    # time_partition, infection_time_series, recover_time_series = sample_sim(A)
    ensemble_res = np.zeros((3, len(time_partition)))

    num_sims = 300
    sims_counted = 0
    for i in range(num_sims):
        if i % 1000 == 0:
            net = nx.erdos_renyi_graph(N, p=beta)
            A = np.array(nx.adjacency_matrix(net).todense())
            adj_list = epi.network.NetworkBuilder.create_adjacency_list(net)
            # A = np.full((N, N), 1)
            # np.fill_diagonal(A, 0)
        tim, inft, rec, exp_ts = sample_sim_seir(A, adj_list)
        # tim, inft, rec = sample_sim(A)
        # only count the epidemics that "take off"
        if inft[20]!=1:
            ensemble_res[0] += inft
            ensemble_res[1] += rec
            ensemble_res[2] += exp_ts
            sims_counted += 1
        # ensemble_res[0] += inft
        # ensemble_res[1] += rec
        # ensemble_res[2] += exp_ts
        # sims_counted += 1
        # ensemble_res[2] += exp_ts
    ensemble_res[0] = ensemble_res[0]/sims_counted
    ensemble_res[1] = ensemble_res[1]/sims_counted
    ensemble_res[2] = ensemble_res[2]/sims_counted
    # plt.plot(time_partition, time_series[1], color='blue', label='Susceptible')
    # plt.plot(time_partition, exposed_time_series/N, color='orange', label='Exposed')
    plt.plot(time_partition, ensemble_res[0]/N, color='red', label='Infected')
    plt.plot(time_partition, ensemble_res[1]/N, color='green', label='Recovered')
    plt.plot(time_partition, ensemble_res[2]/N, color='orange', label='Exposed')
    # plt.ylim([0,1])
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()


# TODO by monday:
# Set up SBM in code to make the sims fit the ODE - turns out, mxing doesn't matter (didn't add mixing in ODE's so its the same)
# still need to address mixing values
# Then fit it to the county data for some county with one example ICE facility (simulate on county data, or model on county data)
# Show that as a monday example

# Prepping for monday meeting:
# 1. At least write a list of the issues with the simulation (need to set up SBM with same shift rates/mixing rates as the ODE)
# 2. Get population data for 1 county (Otay Mesa?) for county and ICE-754
# 3. Get static resident population and use that for facility and staff population number
# 4. Perform ODE model on population numbers
# 5. Compare ODE model results to covid case data for 1. the county 2. the ice cases

# Otay Mesa
# Imperial County, FIPS 06025 (this is the county based on the address in the static dataset, but if you look it up online, looks like it is in 92154, in San Diego county
# Facility ADP 754
# San Diego county pop is 3.1 mil. Imperial county pop is 174,528

# Volunteers could: look for staff population? Number who work there (Vera report used this)

# Adelanto Ice facility
# San Bernardino county 	 FIPS 06071
# population of county:1.7 mil
# facility population ADP: 1330
# 60 staff during a 2 week window, according to this article https://www.desertsun.com/story/news/2020/09/27/114-covid-19-cases-among-detainees-and-staff-adelanto-ice-facility/3555333001/
# article also contains useful counts for the various cities and towns and communities in the desert area which could be extremely helpful for this
# could use mobility data to get where the staff come to/from to correlate them to their "home" towns and get the case data from the Desert sun article
# can someone maybe combine this data into a dataset?
# 81 cases in september, 30 staff cases. see if this fits


