import scipy
from scipy import integrate
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

def example_model(N=1000, show_recovered=False, show_susceptible=False):
    # Example for Adelanto facility in San Bernardino county
    beta = 0.625  # An infected person infects a person every 2 days (half a person per day) (this might be after social distancing so should increase)
    beta = 2.43 # can be greater than 1 for rate in ODE model, adult risk from Lofgren paper
    sigma = 0.5
    gamma = 1/10  # Recover period is 5 days
    gamma_ei = 1/6.7  # latent period is 5 days, say
    staff_work_shift = 3
    c_jail = 3

    county_pop = 1700000
    staff_pop = 60
    detention_pop = 781

    N = detention_pop + county_pop + staff_pop + 10 # 1 and 10 for initial infections, might want to change later

    y_init = np.array([detention_pop/N, 0/N, 0, 0, #detention facility, 1330 ADP but 781 were there in september
                       county_pop/N, 10/N, 0, 0, #community/county
                       (staff_pop/2)/N, 1/N, 0, 0, # staff in community, split up 60 staff on/off shift
                       (staff_pop/2)/N, 0/N, 0, 0]) # staff in facility

    model_params = ModelParams(beta, sigma, gamma, gamma_ei, staff_work_shift, c_jail)
    model_params.callibrate(county_pop, detention_pop, staff_pop, c_0=100000)
    model = Model(model_params, N, y_init)
    solution_ts = model.solve_model()
    print(f'Beta community/detention Params: {model_params.beta_community}, {model_params.beta_detention}')
    t_lim_max = len(solution_ts[0])
    t_lim_max = max(np.where(solution_ts[0]<200)[0])
    plot_ts(solution_ts, t_lim_max, N, county_pop, detention_pop, staff_pop,
            combine_staff=True, show_susceptible=show_susceptible,
            show_recovered=show_recovered, community_separate_plot=True)

    return solution_ts


def solve_and_plot(N=1000, show_recovered=False, show_susceptible=False):
    beta = 0.625  # An infected person infects a person every 2 days (half a person per day)
    sigma = 0.5
    gamma = 1/10  # Recover period is 5 days
    gamma_ei = 1/6.7  # latent period is 5 days, say
    staff_work_shift = 3

    # y_init that might be more realistic
    y_init = np.array([150/N, 0/N, 0, 0, #detention facility
                       799/N, 1/N, 0, 0, #community/county
                       20/N, 0/N, 0, 0, # staff in community
                       30/N, 0/N, 0, 0]) # staff in facility
    # Even split y_init for testing:
    y_init = np.array([250/N, 0/N, 0, 0, #detention facility
                       499/N, 1/N, 0, 0, #community/county
                       125/N, 0/N, 0, 0, # staff in community
                       125/N, 0/N, 0, 0]) # staff in facility
    # For N=100:
    if N==100:
        y_init = np.array([25/N, 0/N, 0, 0, #detention facility
                           49/N, 1/N, 0, 0, #community/county
                           12/N, 0/N, 0, 0, # staff in community
                           13/N, 0/N, 0, 0]) # staff in facility
    model_params = ModelParams(beta, sigma, gamma, gamma_ei, staff_work_shift)
    model = Model(model_params, N, y_init)
    solution_ts = model.solve_model()
    t_lim_max = len(solution_ts[0])
    plot_ts(solution_ts, t_lim_max, N=N, combine_staff=True, show_susceptible=show_susceptible, show_recovered=show_recovered)

    return solution_ts

def plot_ts(time_series, t_lim=15, N=400, county_pop=200, detention_pop=150, staff_pop=50, combine_staff=False, show_susceptible=False, show_recovered=False, community_separate_plot=False):
    time_series = np.array(time_series)
    # time_series = time_series.T
    plt.figure('Detention Cases')
    # Detention
    if show_susceptible:
        plt.plot(time_series[0][:t_lim], time_series[1][:t_lim]*N/detention_pop, color='blue', label='D - Susceptible')
    plt.plot(time_series[0][:t_lim], time_series[2][:t_lim]*N/detention_pop, color='orange', label='D - Exposed')
    plt.plot(time_series[0][:t_lim], time_series[3][:t_lim]*N/detention_pop, color='red', label='D - Infected')
    if show_recovered:
        plt.plot(time_series[0][:t_lim], time_series[4][:t_lim]*N/detention_pop, color='green', label='D - Recovered')

    plt.figure('Staff cases')
    if not combine_staff:
        # Community Staff
        if show_susceptible:
            plt.plot(time_series[0][:t_lim], time_series[9][:t_lim]*N/(staff_pop/2), color='blue', label='C Staff - Susceptible', ls='-.')
        plt.plot(time_series[0][:t_lim], time_series[10][:t_lim]*N/(staff_pop/2), color='orange', label='C Staff - Exposed', ls='-.')
        plt.plot(time_series[0][:t_lim], time_series[11][:t_lim]*N/(staff_pop/2), color='red', label='C Staff - Infected', ls='-.')
        if show_recovered:
            plt.plot(time_series[0][:t_lim], time_series[12][:t_lim]*N/(staff_pop/2), color='green', label='C Staff - Recovered', ls='-.')

        # Detention Staff
        if show_susceptible:
            plt.plot(time_series[0][:t_lim], time_series[13][:t_lim]*N/(staff_pop/2), color='blue', label='D Staff - Susceptible', ls='--')
        plt.plot(time_series[0][:t_lim], time_series[14][:t_lim]*N/(staff_pop/2), color='orange', label='D Staff - Exposed', ls='--')
        plt.plot(time_series[0][:t_lim], time_series[15][:t_lim]*N/(staff_pop/2), color='red', label='D Staff - Infected', ls='--')
        if show_recovered:
            plt.plot(time_series[0][:t_lim], time_series[16][:t_lim]*N/(staff_pop/2), color='green', label='D Staff - Recovered', ls='--')

    # # Staff combined (Comment out specific staff plots above)
    if combine_staff:
        if show_susceptible:
            plt.plot(time_series[0][:t_lim], (time_series[9][:t_lim] + time_series[13][:t_lim])*N/staff_pop, color='blue', label='Staff - Susceptible', ls='-.')
        plt.plot(time_series[0][:t_lim], (time_series[10][:t_lim] + time_series[14][:t_lim])*N/staff_pop, color='orange', label='Staff - Exposed', ls='-.')
        plt.plot(time_series[0][:t_lim], (time_series[11][:t_lim] + time_series[15][:t_lim])*N/staff_pop, color='red', label='Staff - Infected', ls='-.')
        if show_recovered:
            plt.plot(time_series[0][:t_lim], (time_series[12][:t_lim] + time_series[16][:t_lim])*N/staff_pop, color='green', label='Staff - Recovered', ls='-.')

    # plt.xticks(time_series[0], rotation=45, fontsize=10)
    # plt.legend(loc='upper left')
    plt.legend(loc='upper right')
    # plt.ylim([0, 1])
    plt.tight_layout()

    if community_separate_plot:
        plt.figure('community')
    # Community
    if show_susceptible:
        plt.plot(time_series[0][:t_lim], time_series[5][:t_lim]*N/county_pop, color='blue', label='C - Susceptible', ls=':')
    plt.plot(time_series[0][:t_lim], time_series[6][:t_lim]*N/county_pop, color='orange', label='C - Exposed', ls=':')
    plt.plot(time_series[0][:t_lim], time_series[7][:t_lim]*N/county_pop, color='red', label='C - Infected', ls=':')
    if show_recovered:
        plt.plot(time_series[0][:t_lim], time_series[8][:t_lim]*N/county_pop, color='green', label='C - Recovered', ls=':')

    # plt.xticks(time_series[0], rotation=45, fontsize=10)
    # plt.legend(loc='upper left')
    plt.legend(loc='upper right')
    # plt.ylim([0, 1])
    plt.tight_layout()
    # plt.show()


class ModelParams:
    def __init__(self, beta, sigma, gamma, gamma_ei, staff_work_shift, c_jail):
        self.beta = beta
        self.sigma = sigma
        self.gamma = gamma
        self.gamma_ei = gamma_ei

        self.staff_work_shift = staff_work_shift # 1/8 hrs = 1/(1/3)day = 3. Home shift = 1/16 hrs = 1/(2/3)day = 3/2
        self.staff_home_shift = 1/((24 - (1/staff_work_shift)*24)/24)
        self.c_jail = c_jail

    def callibrate(self, county_pop, detention_pop, staff_pop, c_0=1): #unclear whetehr to do staff pop yet
        beta_community_cal = c_0 * self.beta / (county_pop+staff_pop)
        beta_detention_cal = c_0 * self.c_jail * self.beta / (detention_pop+staff_pop)
        # beta_staff_cal = c_0 * self.beta / staff_pop
        self.beta_community = beta_community_cal
        self.beta_detention = beta_detention_cal
        # self.beta_staff = beta_staff_cal



class Model:
    def __init__(self, params, N, y_init):
        self.params = params
        self.N = N
        self.y_init = y_init
        self.time_series = []

    def solve_model(self):
        solution = scipy.integrate.solve_ivp(self.odes_seir_metapop, t_span=[0, 500], y0=self.y_init)
        self.time_series.append(solution.t)
        for i in range(len(self.y_init)):
            self.time_series.append(solution.y[i])
        return self.time_series

    def odes_seir(self, t, y):
        # y form: [s(t), e(t), i(t), r(t)]
        s_t = y[0]
        e_t = y[1]
        i_t = y[2]
        r_t = y[3]
        ds_dt = -self.params.sigma * self.beta * (s_t * e_t) - self.beta * (s_t * i_t)
        de_dt = self.params.sigma * self.beta * (s_t * e_t) + self.beta * (s_t * i_t) - self.gamma_ei * e_t
        di_dt = -self.gamma * i_t + self.gamma_ei * e_t
        dr_dt = self.gamma * i_t
        new_y = np.array([ds_dt, de_dt, di_dt, dr_dt])
        return new_y

    def odes_seir_metapop(self, t, y):
        # y form: [s(t), e(t), i(t), r(t)]

        # Detention residents
        s_t_D = y[0]
        e_t_D = y[1]
        i_t_D = y[2]
        r_t_D = y[3]

        # Community
        s_t_C = y[4]
        e_t_C = y[5]
        i_t_C = y[6]
        r_t_C = y[7]

        # Employees in the commmunity
        s_t_O_C = y[8]
        e_t_O_C = y[9]
        i_t_O_C = y[10]
        r_t_O_C = y[11]

        # Employees in the facility
        s_t_O_D = y[12]
        e_t_O_D = y[13]
        i_t_O_D = y[14]
        r_t_O_D = y[15]

        print(sum(y))

        # todo using same beta values for every combo right now, and same mixing rates for staff/detainees as detainees w each other
        ds_dt_D = - self.params.sigma * self.params.beta_detention * (s_t_D * e_t_D) - self.params.beta_detention * (s_t_D * i_t_D) \
                  - self.params.sigma * self.params.beta_detention * (s_t_D * e_t_O_D) - self.params.beta_detention * (s_t_D * i_t_O_D)
        de_dt_D = self.params.sigma * self.params.beta_detention * (s_t_D * e_t_D) + self.params.beta_detention * (s_t_D * i_t_D) - self.params.gamma_ei * e_t_D \
                    + self.params.sigma * self.params.beta_detention * (s_t_D * e_t_O_D) + self.params.beta_detention * (s_t_D * i_t_O_D)
        di_dt_D = -self.params.gamma * i_t_D + self.params.gamma_ei * e_t_D
        dr_dt_D = self.params.gamma * i_t_D

        # todo make comm/staff specific beta?
        ds_dt_C = -self.params.sigma * self.params.beta_community * (s_t_C * e_t_C) - self.params.beta_community * (s_t_C * i_t_C) \
                    -self.params.sigma * self.params.beta_community * (s_t_C * e_t_O_C) - self.params.beta_community * (s_t_C * i_t_O_C)
        de_dt_C = self.params.sigma * self.params.beta_community * (s_t_C * e_t_C) + self.params.beta_community * (s_t_C * i_t_C) - self.params.gamma_ei * e_t_C \
                  + self.params.sigma * self.params.beta_community * (s_t_C * e_t_O_C) + self.params.beta_community * (s_t_C * i_t_O_C)
        di_dt_C = -self.params.gamma * i_t_C + self.params.gamma_ei * e_t_C
        dr_dt_C = self.params.gamma * i_t_C

        # todo don't think staff should mix with staff while "at home", just have the flux related to community mixing
        #  so e_t_O_C for example should be the susceptible pop mixing with the exposed/infected pop of community
        ds_dt_O_C = -self.params.sigma * self.params.beta_community * (s_t_O_C * e_t_C) - self.params.beta_community * (s_t_O_C * i_t_C) \
                  + self.params.staff_work_shift * s_t_O_D - self.params.staff_home_shift * s_t_O_C
        de_dt_O_C = self.params.sigma * self.params.beta_community * (s_t_O_C * e_t_C) + self.params.beta_community * (s_t_O_C * i_t_C) - self.params.gamma_ei * e_t_O_C \
                    + self.params.staff_work_shift * e_t_O_D - self.params.staff_home_shift * e_t_O_C
        di_dt_O_C = -self.params.gamma * i_t_O_C + self.params.gamma_ei * e_t_O_C \
                    + self.params.staff_work_shift * i_t_O_D - self.params.staff_home_shift * i_t_O_C
        dr_dt_O_C = self.params.gamma * i_t_O_C \
                    + self.params.staff_work_shift * r_t_O_D - self.params.staff_home_shift * r_t_O_C

        ds_dt_O_D = -self.params.sigma * self.params.beta_detention * (s_t_O_D * e_t_O_D) - self.params.beta_detention * (s_t_O_D * i_t_O_D) \
                    -self.params.sigma * self.params.beta_detention * (s_t_O_D * e_t_D) - self.params.beta_detention * (s_t_O_D * i_t_D) \
                    - self.params.staff_work_shift * s_t_O_D + self.params.staff_home_shift * s_t_O_C
        de_dt_O_D = self.params.sigma * self.params.beta_detention * (s_t_O_D * e_t_O_D) + self.params.beta_detention * (s_t_O_D * i_t_O_D) - self.params.gamma_ei * e_t_O_D \
                    + self.params.sigma * self.params.beta_detention * (s_t_O_D * e_t_D) + self.params.beta_detention * (s_t_O_D * i_t_D) \
                    - self.params.staff_work_shift * e_t_O_D + self.params.staff_home_shift * e_t_O_C
        di_dt_O_D = -self.params.gamma * i_t_O_D + self.params.gamma_ei * e_t_O_D \
                    - self.params.staff_work_shift * i_t_O_D + self.params.staff_home_shift * i_t_O_C
        dr_dt_O_D = self.params.gamma * i_t_O_D \
                    - self.params.staff_work_shift * r_t_O_D + self.params.staff_home_shift * r_t_O_C

        new_y = np.array([ds_dt_D, de_dt_D, di_dt_D, dr_dt_D,
                          ds_dt_C, de_dt_C, di_dt_C, dr_dt_C,
                          ds_dt_O_C, de_dt_O_C, di_dt_O_C, dr_dt_O_C,
                          ds_dt_O_D, de_dt_O_D, di_dt_O_D, dr_dt_O_D])
        return new_y


if __name__ == '__main__':
    example_model()
    solve_and_plot()
