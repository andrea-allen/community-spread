import scipy
from scipy import integrate
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import epintervene.simobjects as epi
# TODO next: Inferring the right transmission probabilities for the simulation from the rate parameters

t_res = []
time_series = []
beta = 0.9# An infected person infects a person every 2 days (half a person per day)
sigma = 0.5
gamma = 0.2 #Recover period is 5 days
gamma_ei = 0.2 # latent period is 5 days, say
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
    print(t)
    # t_res.append(t)
    # time_series.append([t, s_t, i_t, r_t])
    print([s_t+ds_dt, i_t+di_dt, r_t+dr_dt])
    # plt.scatter(t, s_t, color='blue')
    # plt.scatter(t, e_t, color='orange')
    # plt.scatter(t, i_t, color='red')
    # plt.scatter(t, r_t, color='green')
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

def sample_sim_seir(A):
    Beta = np.full((N, N), beta/N)
    np.fill_diagonal(Beta, 0)
    Beta_SE = np.full((N, N), sigma*beta/N)
    np.fill_diagonal(Beta_SE, 0)
    Gamma_EI = np.full((N), gamma_ei) #TODO does this end up sounding right for the model, if you make gamma 1 for ex. then it ends up as 1 per day
    Gamma = np.full((N), gamma)
    seir_sim = epi.SimulationSEIR(A)
    # seir_sim = epi.Simulation(A)
    seir_sim.add_infection_event_rates(Beta)
    seir_sim.add_exposed_event_rates(Beta_SE)
    seir_sim.add_exposed_infected_event_rates(Gamma_EI)
    seir_sim.add_recover_event_rates(Gamma)
    seir_sim.run_sim(wait_for_recovery=True)
    return seir_sim.tabulate_continuous_time(custom_range=True, custom_t_lim=45)

if __name__ == '__main__':
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
    # net = nx.erdos_renyi_graph(N, p=1.0)
    # A = np.array(nx.adjacency_matrix(net).todense())
    A = np.full((N, N), 1)
    np.fill_diagonal(A, 0)
    time_partition, infection_time_series, recover_time_series, exposed_time_series = sample_sim_seir(A)
    # time_partition, infection_time_series, recover_time_series = sample_sim(A)
    ensemble_res = np.zeros((3, len(time_partition)))

    num_sims = 30
    sims_counted = 0
    for i in range(num_sims):
        if i % 10 == 0:
            # net = nx.erdos_renyi_graph(N, p=1.0)
            # A = np.array(nx.adjacency_matrix(net).todense())
            A = np.full((N, N), 1)
            np.fill_diagonal(A, 0)
        tim, inft, rec, exp_ts = sample_sim_seir(A)
        # tim, inft, rec = sample_sim(A)
        if inft[30]!=1:
            ensemble_res[0] += inft/N
            ensemble_res[1] += rec/N
            ensemble_res[2] += exp_ts/N
            sims_counted += 1
        # ensemble_res[2] += exp_ts
    ensemble_res[0] = ensemble_res[0]/sims_counted
    ensemble_res[1] = ensemble_res[1]/sims_counted
    ensemble_res[2] = ensemble_res[2]/sims_counted
    # plt.plot(time_partition, time_series[1], color='blue', label='Susceptible')
    # plt.plot(time_partition, exposed_time_series/N, color='orange', label='Exposed')
    plt.plot(time_partition, ensemble_res[0], color='red', label='Infected')
    plt.plot(time_partition, ensemble_res[1], color='green', label='Recovered')
    plt.plot(time_partition, ensemble_res[2], color='orange', label='Exposed')
    plt.ylim([0,1])
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

