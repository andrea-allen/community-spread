import scipy
from scipy import integrate
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx


def example_model(county_pop=10000, staff_pop=10, detention_pop=100, show_recovered=False, show_susceptible=False,
                  beta=2.43, sigma=0.5, gamma=1 / 10, gamma_ei=1 / 6.7, staff_work_shift=3, c_jail=3, c_0=500,
                  init_community_infections=500, init_detention_infections=0, arrest_rate=.0000035, alos = .0167,
                  normalize=True, same_plot=False, num_days=60, model_params=None):
    # Should be able to set parameters with county (city) and ICE facility and staff population
    # Example for Adelanto facility in San Bernardino county
    # Defaults:
    # beta = 0.625  # An infected person infects a person every 2 days (half a person per day) (this might be after social distancing so should increase)
    # beta = 2.43 # can be greater than 1 for rate in ODE model, adult risk from Lofgren paper
    # sigma = 0.5
    # gamma = 1/10  # Recover period is 5 days
    # gamma_ei = 1/6.7  # latent period is 5 days, say
    # staff_work_shift = 3
    # c_jail = 3

    # county_pop = 1700000
    # staff_pop = 60
    # detention_pop = 781

    N = model_params.detention_pop + model_params.county_pop + model_params.staff_pop + model_params.init_community_infections + model_params.init_detention_infections  # 1 and 10 for initial infections, might want to change later
    # according to the adelanto data, 4800 cumulative infections by 9/1 (I think) so use however many of those are active to start?

    y_init = np.array([(model_params.detention_pop - model_params.init_detention_infections) / N, 0.5 * model_params.init_detention_infections / N,
                       0.5 * model_params.init_detention_infections / N, 0,
                       # detention facility, 1330 ADP but 781 were there in september
                       model_params.county_pop / N, 0.5 * model_params.init_community_infections / N, 0.5 * model_params.init_community_infections / N, 0,
                       # community/county
                       (model_params.staff_pop / 2) / N, 1 / N, 0, 0,  # staff in community, split up 60 staff on/off shift
                       (model_params.staff_pop / 2) / N, 0 / N, 0, 0])  # staff in facility

    if model_params is None:
        model_params = ModelParams(beta, sigma, gamma, gamma_ei, staff_work_shift, c_jail, arrest_rate, alos,
                                   county_pop, staff_pop, detention_pop, c_0, init_community_infections,
                                   init_detention_infections)
    model_params.callibrate()
    model = Model(model_params, N, y_init)
    solution_ts = model.solve_model()
    print(f'Beta community/detention Params: {model_params.beta_community}, {model_params.beta_detention}')
    t_lim_max = len(solution_ts[0])
    t_lim_max = max(np.where(solution_ts[0] < num_days)[0])
    plot_ts(solution_ts, t_lim_max, N, model_params.county_pop, model_params.detention_pop, model_params.staff_pop,
            combine_staff=True, show_susceptible=show_susceptible,
            show_recovered=show_recovered, community_separate_plot=True, normalize=normalize, same_plot=same_plot)
    # plt.show()

    return solution_ts


def solve_and_plot(N=1000, show_recovered=False, show_susceptible=False):
    beta = 0.625  # An infected person infects a person every 2 days (half a person per day)
    sigma = 0.5
    gamma = 1 / 10  # Recover period is 5 days
    gamma_ei = 1 / 6.7  # latent period is 5 days, say
    staff_work_shift = 3

    # y_init that might be more realistic
    y_init = np.array([150 / N, 0 / N, 0, 0,  # detention facility
                       799 / N, 1 / N, 0, 0,  # community/county
                       20 / N, 0 / N, 0, 0,  # staff in community
                       30 / N, 0 / N, 0, 0])  # staff in facility
    # Even split y_init for testing:
    y_init = np.array([250 / N, 0 / N, 0, 0,  # detention facility
                       499 / N, 1 / N, 0, 0,  # community/county
                       125 / N, 0 / N, 0, 0,  # staff in community
                       125 / N, 0 / N, 0, 0])  # staff in facility
    # For N=100:
    if N == 100:
        y_init = np.array([25 / N, 0 / N, 0, 0,  # detention facility
                           49 / N, 1 / N, 0, 0,  # community/county
                           12 / N, 0 / N, 0, 0,  # staff in community
                           13 / N, 0 / N, 0, 0])  # staff in facility
    model_params = ModelParams(beta, sigma, gamma, gamma_ei, staff_work_shift)
    model = Model(model_params, N, y_init)
    solution_ts = model.solve_model()
    t_lim_max = len(solution_ts[0])
    plot_ts(solution_ts, t_lim_max, N=N, combine_staff=True, show_susceptible=show_susceptible,
            show_recovered=show_recovered)

    return solution_ts


def plot_ts(time_series, t_lim=15, N=400, county_pop=200, detention_pop=150, staff_pop=50, combine_staff=False,
            show_susceptible=False, show_recovered=False, community_separate_plot=False, normalize=False,
            same_plot=False):
    more_colors = [ "#D7790F", "#82CAA4", "#4C6788", "#84816F", "#71A9C9", "#AE91A8"]
    # Also would be good to have a way to interpolate this into incidence data to match reports
    time_series = np.array(time_series)
    # time_series = time_series.T
    if not same_plot:
        plt.figure('Detention Cases')
    plt.xlabel('Days')
    plt.ylabel('Percent of population')
    # Detention
    if not normalize:
        detention_pop = 1
    if show_susceptible:
        plt.plot(time_series[0][:t_lim], time_series[1][:t_lim] * N / detention_pop, color='blue',
                 label='D - Susceptible')
    plt.plot(time_series[0][:t_lim], time_series[2][:t_lim] * N / detention_pop, color=more_colors[1], label='D - Exposed')
    plt.plot(time_series[0][:t_lim], time_series[3][:t_lim] * N / detention_pop, color=more_colors[0], label='D - Infected')
    if show_recovered:
        plt.plot(time_series[0][:t_lim], time_series[4][:t_lim] * N / detention_pop, color='green',
                 label='D - Recovered')

    if not same_plot:
        plt.figure('Staff cases')
    plt.xlabel('Days')
    plt.ylabel('Percent of population')
    if not combine_staff:
        # Community Staff
        if not normalize:
            staff_pop = 2
        if show_susceptible:
            plt.plot(time_series[0][:t_lim], time_series[9][:t_lim] * N / (staff_pop / 2), color='blue',
                     label='C Staff - Susceptible', ls='-.')
        plt.plot(time_series[0][:t_lim], time_series[10][:t_lim] * N / (staff_pop / 2), color=more_colors[1],
                 label='C Staff - Exposed', ls='-.')
        plt.plot(time_series[0][:t_lim], time_series[11][:t_lim] * N / (staff_pop / 2), color=more_colors[0],
                 label='C Staff - Infected', ls='-.')
        if show_recovered:
            plt.plot(time_series[0][:t_lim], time_series[12][:t_lim] * N / (staff_pop / 2), color='green',
                     label='C Staff - Recovered', ls='-.')

        # Detention Staff
        if show_susceptible:
            plt.plot(time_series[0][:t_lim], time_series[13][:t_lim] * N / (staff_pop / 2), color='blue',
                     label='D Staff - Susceptible', ls='--')
        plt.plot(time_series[0][:t_lim], time_series[14][:t_lim] * N / (staff_pop / 2), color=more_colors[1],
                 label='D Staff - Exposed', ls='--')
        plt.plot(time_series[0][:t_lim], time_series[15][:t_lim] * N / (staff_pop / 2), color=more_colors[0],
                 label='D Staff - Infected', ls='--')
        if show_recovered:
            plt.plot(time_series[0][:t_lim], time_series[16][:t_lim] * N / (staff_pop / 2), color='green',
                     label='D Staff - Recovered', ls='--')

    # # Staff combined (Comment out specific staff plots above)
    if combine_staff:
        if not normalize:
            staff_pop = 1
        if show_susceptible:
            plt.plot(time_series[0][:t_lim], (time_series[9][:t_lim] + time_series[13][:t_lim]) * N / staff_pop,
                     color='blue', label='Staff - Susceptible', ls='-.')
        plt.plot(time_series[0][:t_lim], (time_series[10][:t_lim] + time_series[14][:t_lim]) * N / staff_pop,
                 color=more_colors[1], label='Staff - Exposed', ls='-.')
        plt.plot(time_series[0][:t_lim], (time_series[11][:t_lim] + time_series[15][:t_lim]) * N / staff_pop,
                 color=more_colors[0], label='Staff - Infected', ls='-.')
        if show_recovered:
            plt.plot(time_series[0][:t_lim], (time_series[12][:t_lim] + time_series[16][:t_lim]) * N / staff_pop,
                     color='green', label='Staff - Recovered', ls='-.')

    # plt.xticks(time_series[0], rotation=45, fontsize=10)
    plt.yticks([10**(-3), 10**(-2), 10**(-1)], ['10', '100', '1,000'])
    plt.ylabel('Cases per 10,000 People')
    # plt.legend(loc='upper left')
    plt.legend(loc='upper right')
    #TODO: cases per 10k instead?
    # plt.ylim([0, 1])
    plt.tight_layout()

    if not same_plot:
        if community_separate_plot:
            plt.figure('community')
            plt.xlabel('Days')
            plt.ylabel('Percent of population')
    # Community
    if not normalize:
        county_pop = 1
    if show_susceptible:
        plt.plot(time_series[0][:t_lim], time_series[5][:t_lim] * N / county_pop, color='blue', label='C - Susceptible',
                 ls=':')
    plt.plot(time_series[0][:t_lim], time_series[6][:t_lim] * N / county_pop, color=more_colors[1], label='C - Exposed',
             ls=':')
    plt.plot(time_series[0][:t_lim], time_series[7][:t_lim] * N / county_pop, color=more_colors[0], label='C - Infected', ls=':')
    if show_recovered:
        plt.plot(time_series[0][:t_lim], time_series[8][:t_lim] * N / county_pop, color='green', label='C - Recovered',
                 ls=':')

    # plt.xticks(time_series[0], rotation=45, fontsize=10)
    # plt.legend(loc='upper left')
    plt.legend(loc='upper right')
    # plt.ylim([0, 1])
    plt.tight_layout()
    # plt.show()


class ModelParams:
    def __init__(self, beta, sigma, gamma, gamma_ei, staff_work_shift, c_jail, arrest_rate, alos,
                 county_pop, staff_pop, detention_pop, c_0, init_community_infections,
                 init_detention_infections):
        self.beta = beta
        self.sigma = sigma
        self.gamma = gamma
        self.gamma_ei = gamma_ei

        self.staff_work_shift = staff_work_shift  # 1/8 hrs = 1/(1/3)day = 3. Home shift = 1/16 hrs = 1/(2/3)day = 3/2
        self.staff_home_shift = 1 / ((24 - (1 / staff_work_shift) * 24) / 24)
        self.c_jail = c_jail
        self.c_0 = c_0
        self.arrest_rate = arrest_rate
        self.alos = alos

        self.county_pop = county_pop
        self.staff_pop = staff_pop
        self.detention_pop = detention_pop

        self.init_community_infections = init_community_infections
        self.init_detention_infections = init_detention_infections

    def callibrate(self):  # unclear whetehr to do staff pop yet
        beta_community_cal = self.c_0 * self.beta / (self.county_pop + self.staff_pop)
        beta_detention_cal = self.c_0 * self.c_jail * self.beta / (self.detention_pop + self.staff_pop)
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

        # detention
        ds_dt_D = - self.params.sigma * self.params.beta_detention * (s_t_D * e_t_D) - self.params.beta_detention * (
                    s_t_D * i_t_D) \
                  - self.params.sigma * self.params.beta_detention * (s_t_D * e_t_O_D) - self.params.beta_detention * (
                              s_t_D * i_t_O_D) \
                  + self.params.arrest_rate * s_t_C
        de_dt_D = self.params.sigma * self.params.beta_detention * (s_t_D * e_t_D) + self.params.beta_detention * (
                    s_t_D * i_t_D) - self.params.gamma_ei * e_t_D \
                  + self.params.sigma * self.params.beta_detention * (s_t_D * e_t_O_D) + self.params.beta_detention * (
                              s_t_D * i_t_O_D)
        di_dt_D = -self.params.gamma * i_t_D + self.params.gamma_ei * e_t_D \
                  - self.params.alos * i_t_D
        dr_dt_D = self.params.gamma * i_t_D

        # community
        ds_dt_C = -self.params.sigma * self.params.beta_community * (s_t_C * e_t_C) - self.params.beta_community * (
                    s_t_C * i_t_C) \
                  - self.params.sigma * self.params.beta_community * (s_t_C * e_t_O_C) - self.params.beta_community * (
                              s_t_C * i_t_O_C) \
                  - self.params.arrest_rate * s_t_C
        de_dt_C = self.params.sigma * self.params.beta_community * (s_t_C * e_t_C) + self.params.beta_community * (
                    s_t_C * i_t_C) - self.params.gamma_ei * e_t_C \
                  + self.params.sigma * self.params.beta_community * (s_t_C * e_t_O_C) + self.params.beta_community * (
                              s_t_C * i_t_O_C)
        di_dt_C = -self.params.gamma * i_t_C + self.params.gamma_ei * e_t_C \
                  + self.params.alos * i_t_D
        dr_dt_C = self.params.gamma * i_t_C

        # employees/officers
        ds_dt_O_C = -self.params.sigma * self.params.beta_community * (s_t_O_C * e_t_C) - self.params.beta_community * (
                    s_t_O_C * i_t_C) \
                    + self.params.staff_work_shift * s_t_O_D - self.params.staff_home_shift * s_t_O_C
        de_dt_O_C = self.params.sigma * self.params.beta_community * (s_t_O_C * e_t_C) + self.params.beta_community * (
                    s_t_O_C * i_t_C) - self.params.gamma_ei * e_t_O_C \
                    + self.params.staff_work_shift * e_t_O_D - self.params.staff_home_shift * e_t_O_C
        di_dt_O_C = -self.params.gamma * i_t_O_C + self.params.gamma_ei * e_t_O_C \
                    + self.params.staff_work_shift * i_t_O_D - self.params.staff_home_shift * i_t_O_C
        dr_dt_O_C = self.params.gamma * i_t_O_C \
                    + self.params.staff_work_shift * r_t_O_D - self.params.staff_home_shift * r_t_O_C

        ds_dt_O_D = -self.params.sigma * self.params.beta_detention * (
                    s_t_O_D * e_t_O_D) - self.params.beta_detention * (s_t_O_D * i_t_O_D) \
                    - self.params.sigma * self.params.beta_detention * (
                                s_t_O_D * e_t_D) - self.params.beta_detention * (s_t_O_D * i_t_D) \
                    - self.params.staff_work_shift * s_t_O_D + self.params.staff_home_shift * s_t_O_C
        de_dt_O_D = self.params.sigma * self.params.beta_detention * (
                    s_t_O_D * e_t_O_D) + self.params.beta_detention * (
                                s_t_O_D * i_t_O_D) - self.params.gamma_ei * e_t_O_D \
                    + self.params.sigma * self.params.beta_detention * (
                                s_t_O_D * e_t_D) + self.params.beta_detention * (s_t_O_D * i_t_D) \
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
    # county_pop = 1700000
    # staff_pop = 60
    # detention_pop = 781
    # San Bernardino county example:
    # example_model(county_pop=1700000, staff_pop=60, detention_pop=781)
    # Adams County Mississippi example: (guessing staff pop)
    # 2407 initial cases in march
    # Can observe c_jail from real data, comparing the per-10000 person rate. Then show a model which makes for a compelling argument
    # that even if spread is dying down in the community, it can still have an affect on within-detention case rates.
    # example_model(county_pop=30693, c_jail=160, staff_pop=50, detention_pop=338, init_community_infections=2200, show_recovered=True)
    # ICE SOUTH TEXAS DETENTION COMPLEX: params inferred from data, avg c_jail, should get to 2700 infections after 60 days
    # initial detention infections is 300, should get to 400 to match reports
    example_model(county_pop=20306, staff_pop=100, detention_pop=650, show_recovered=False, show_susceptible=False,
                  beta=2.43,sigma=0.5, gamma=1 / 10, gamma_ei=1 / 6.7, staff_work_shift=3, c_jail=17, c_0=500,
                  init_community_infections=244, init_detention_infections=0,arrest_rate=.00067, alos=.033,
                  normalize=True, same_plot=True)
    # This model is good for Frio county. Do a write up of why ^
    # solve_and_plot(show_recovered=True)
