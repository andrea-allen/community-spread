import pandas as pd
import matplotlib.pyplot as plt
import folium
import numpy as np
from pandas.io.json import json_normalize
import json
import datetime
from coviddata import get_sandiego_api_covid
from diseasemodel import model


def load_nyt_data():
    nyt_url_counties = 'https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv'
    df = pd.read_csv(nyt_url_counties, na_filter=True)
    return df


def load_ucla_data(fname=None):
    if fname is not None:
        data = pd.read_csv(fname)
        return data
    else:
        data = pd.read_csv("http://104.131.72.50:3838/scraper_data/summary_data/scraped_time_series.csv")
        return data


def select_ice_facilities(data):
    ice_data = data[data['Jurisdiction'].str.contains('immigration', na=False)]
    print(ice_data.head(10))
    return ice_data


def select_california_ice(data):
    ice_data = data[data['State'].str.contains('California', na=False)]
    print(ice_data.head(10))
    return ice_data


def hospital_data():
    hospital_df = pd.read_csv('./../coviddata/covid19hospitalbycounty.csv')
    df_head = hospital_df.head(5)
    outbreak_df = pd.read_csv('./../coviddata/covid-19_outbreaks_032221.csv')
    print(outbreak_df.head())
    print(df_head)


def moving_avg(covid_data, days=7):
    ma_df = covid_data.rolling(days).obj
    plt.plot(ma_df)
    plt.show()


def county_facility_x_correlation(facility, county, start_date, end_date, facility_name, county_pop):
    county_name = county.head(1)['county'].values[0]
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    facility['Date'] = pd.to_datetime(facility['Date'])
    facility_mask = (facility['Date'] > start_date) & (facility['Date'] <= end_date)
    facility = facility.loc[facility_mask]

    county['date'] = pd.to_datetime(county['date'])
    county_mask = (county['date'] > start_date) & (county['date'] <= end_date)
    county = county.loc[county_mask]

    plt.plot(facility['Residents.Confirmed'])
    plt.xlabel('Days')
    plt.ylabel('Cumulative case count')
    plt.show()

    plt.plot(county['cases'])
    plt.xlabel('Days')
    plt.ylabel('Cumulative case count')
    plt.show()

    # plt.figure('facility')
    # facility['Rolling_diff'] = moving_average(np.array(facility['Residents.Confirmed'].diff(1))[1:])
    # # plt.plot(facility['Date'], facility['Residents.Confirmed'].diff(1), color='blue')
    # plt.plot(facility['Date'], facility['Rolling_diff'], color='blue')
    # plt.xticks(rotation=45)
    # plt.xlabel('Days')
    # plt.ylabel('Cumulative case count')
    # plt.title(f'7 Day Rolling Avg - {facility_name}')
    # plt.show()
    #
    # plt.figure('county')
    # county['Rolling_diff'] = moving_average(np.array(county['cases'].diff(1))[1:])
    # # plt.plot(county['date'], county['cases'].diff(1), color='orange')
    # plt.plot(county['date'], county['Rolling_diff'], color='orange')
    # plt.xticks(rotation=45)
    # plt.xlabel('Days')
    # plt.ylabel('Cumulative case count')
    # plt.title(f'7 Day Rolling Avg - County {county_name}')
    # plt.show()

    joined_df = county.join(facility.set_index('Date'), on='date', how='left')

    ## TODO before doing the correlation, need to join on the date column to get the same date values for NYT and ICE data
    ## basically need to build up some more of my data tools first
    # Compute rolling window synchrony
    d1 = joined_df['Residents.Active'].fillna(method='ffill').dropna()[1:] / 338 * 10000
    d2 = joined_df['cases'].fillna(method='ffill').dropna()[1:] / 30000 * 10000
    rs = np.array([crosscorr(d1, d2, lag) for lag in range(-min(len(joined_df),21), min(len(joined_df),21))]) #21 days
    rs_not_nan = rs[~ np.isnan(rs)]
    offset = np.floor(len(rs) / 2) - np.nanargmax(rs)
    f, ax = plt.subplots(figsize=(14, 5))
    ax.plot(rs)
    ax.axvline(np.ceil(len(rs) / 2), color='k', linestyle='--', label='Center')
    ax.axvline(np.nanargmax(rs), color='r', linestyle='--', label='Peak synchrony')
    ax.set(title=f'{facility_name} cross correlation with {county_name} county \n Date Offset = {offset} frames',
           xlabel='Offset',
           ylabel='Pearson r')
    # ax.set_xticks(np.arange(len(joined_df)))
    # ax.set_xticklabels(joined_df['Date'])
    plt.legend()

    plt.figure('compare case rates')
    avg_difference_in_rates = np.average(joined_df['Residents.Active'].fillna(method='ffill')[1:] / facility.head(1)['Population.Feb20'].values[0] * 10000)/np.average(joined_df['cases'].diff(10).fillna(method='bfill') / county_pop * 10000)
    plt.ylabel('Active case rate per 10,000 people')
    plt.title(f'Active case rates for {facility_name} and surrounding county\n'
              f'Avg rate of detention facility is {np.round(avg_difference_in_rates,1)}X higher than county rate')
    plt.plot(joined_df['date'], joined_df['Residents.Active'].fillna(method='ffill') / facility.head(1)['Population.Feb20'].values[0] * 10000, label=f'{facility_name} Detainee Rate')
    plt.plot(joined_df['date'], joined_df['cases'].diff(10).fillna(method='bfill') / county_pop * 10000, label='County rate')
    plt.xticks(rotation=45)
    plt.ylim(1, 100000)
    plt.semilogy()
    plt.yticks([10, 100, 1000, 10000], labels=['10', '100', '1000', '10000'])
    plt.legend(loc='upper left')
    plt.show()


def crosscorr(datax, datay, lag=0, wrap=False):
    """ Lag-N cross correlation.
    Shifted data filled with NaNs

    Parameters
    ----------
    lag : int, default 0
    datax, datay : pandas.Series objects of equal length
    Returns
    ----------
    crosscorr : float
    """
    if wrap:
        shiftedy = datay.shift(lag)
        shiftedy.iloc[:lag] = datay.iloc[-lag:].values
        return datax.corr(shiftedy)
    else:
        return datax.corr(datay.shift(lag))


### FOR VOLUNTEER: Use this method as a baseline, goal is:
###  Have either a ref table or an additional column to the main dataset (or column in the static ICE data)
###  that is, for each individual facility, the number of total facilities encompassed in the same county as that
###  facility.
def facilities_per_area():
    # This method will achieve what we want for the chloropleth map by adding a column per county FIPS code of the
    # number of total faciltiies, detention facilities, and jail/prison facilities per county. To make this
    # dataset permanent, feel free to manipulate it however and put it into a more permanent data structure that can
    # be persisted somewhere.

    ucla_data = pd.read_csv("http://104.131.72.50:3838/scraper_data/summary_data/scraped_time_series.csv")
    # Getting county data
    county_name_to_fips = pd.read_csv(
        "https://raw.githubusercontent.com/kjhealy/fips-codes/master/state_and_county_fips_master.csv", header=0,
        delimiter=',')
    state_name_to_abbrv = pd.read_csv("coviddata/name-abbr.csv")
    state_name_to_abbrv['NAME'] = state_name_to_abbrv['NAME'].str.lower()
    county_name_to_fips['name'] = county_name_to_fips['name'].str.lower()

    # Grouping data we have by facility ID and taking one representative row per group
    facility_ids = ucla_data.groupby('Facility.ID').first()
    facility_ids['County'] = facility_ids['County'].str.lower()
    facility_ids['State'] = facility_ids['State'].str.lower()
    cleaned_df = pd.DataFrame(facility_ids[['State', 'County', 'Jurisdiction']])

    # Join facility IDs with FIPS for their respective county
    cleaned_df = cleaned_df.join(state_name_to_abbrv.set_index('NAME'), on='State')
    cleaned_df['county_appended'] = cleaned_df['County'] + ' county'
    cleaned_df = cleaned_df.join(county_name_to_fips.set_index(['state', 'name']), on=[' ABBRV', 'county_appended'])

    # Split into ICE / other facilities and get counts per fips code
    other_facilities = cleaned_df[~cleaned_df['Jurisdiction'].str.contains('immigration')]
    ice_facilities = cleaned_df[cleaned_df['Jurisdiction'].str.contains('immigration', na=False)]
    prisons_jails_per_fips = other_facilities.groupby('fips').count()[['State']]
    ice_per_fips = ice_facilities.groupby('fips').count()[['State']]
    combined_df = prisons_jails_per_fips.join(ice_per_fips, how='outer', lsuffix='_other', rsuffix='_ice').fillna(0)

    # Clean up, make a total count column
    combined_df['fips'] = combined_df.index.astype(int)
    combined_df['fips'] = combined_df['fips'].astype(str)
    combined_df.rename(columns={'State_other': 'count_other'}, inplace=True)
    combined_df.rename(columns={'State_ice': 'count_ice'}, inplace=True)
    combined_df['count_other'] = combined_df.astype(int)
    combined_df['count_ice'] = combined_df.astype(int)
    combined_df['count_total'] = combined_df['count_other'] + combined_df['count_ice']

    return combined_df


### Using the facilities_per_area result, can make a chloropleth map of facility count per county to get some
###  visual cues to guide our intuition when looking at community spread.
def facility_density_map():
    combined_df = facilities_per_area()
    # Create map object (second column denotes which column to use for the values)
    m = folium.Map(location=[41, -97], zoom_start=5)
    us_counties = 'https://gist.githubusercontent.com/wrobstory/5586482/raw/6031540596a4ff6cbfee13a5fc894588422fd3e6/us-counties.json'
    folium.Choropleth(
        geo_data=us_counties,
        name='choropleth',
        data=combined_df,
        columns=['fips', 'count_total'],
        key_on='feature.id',
        fill_color='Reds',
        fill_opacity=0.8,
        nan_fill_opacity=0.0,
        line_opacity=1
    ).add_to(m)
    m.save('count_total.html')

    m = folium.Map(location=[41, -97], zoom_start=5)
    us_counties = 'https://gist.githubusercontent.com/wrobstory/5586482/raw/6031540596a4ff6cbfee13a5fc894588422fd3e6/us-counties.json'
    folium.Choropleth(
        geo_data=us_counties,
        name='choropleth',
        data=combined_df,
        columns=['fips', 'count_other'],
        key_on='feature.id',
        fill_color='Oranges',
        fill_opacity=0.8,
        nan_fill_opacity=0.0,
        line_opacity=1
    ).add_to(m)
    m.save('count_other.html')

    m = folium.Map(location=[41, -97], zoom_start=5)
    us_counties = 'https://gist.githubusercontent.com/wrobstory/5586482/raw/6031540596a4ff6cbfee13a5fc894588422fd3e6/us-counties.json'
    folium.Choropleth(
        geo_data=us_counties,
        name='choropleth',
        data=combined_df,
        columns=['fips', 'count_ice'],
        key_on='feature.id',
        fill_color='Blues',
        fill_opacity=0.8,
        nan_fill_opacity=0.0,
        line_opacity=1
    ).add_to(m)
    m.save('count_ice.html')


# County vs Facility time series correlation for each facility and its county in California
def ca_ice_cross_correlation():
    ucla_Data = load_ucla_data()
    ice_facilities = select_ice_facilities(ucla_Data)
    ca_facilities = select_california_ice(ice_facilities)
    start_date = '2020-12-01'
    end_date = '2021-04-03'
    ca_facility_names = np.unique(ca_facilities[['Name']])
    county_dict = {'ADELANTO ICE PROCESSING CENTER': 'San Bernardino', 'ICE GOLDEN STATE ANNEX FACILITY': 'Kern',
                   'ICE IMPERIAL REGIONAL DETENTION FACILITY': 'Imperial', 'ICE MESA VERDE DETENTION FACILITY': 'Kern',
                   'ICE LA STAGING': 'Los Angeles',
                   'ICE OTAY MESA DETENTION CENTER SAN DIEGO CDF': 'San Diego',
                   'ICE YUBA COUNTY JAIL': 'Yuba'
                   }

    # TODO focus on OTAY MESA, using data from san diego county

    nyt_data = load_nyt_data()
    for facility_name in ca_facility_names:
        facility = ca_facilities[ca_facilities['Name'].str.contains(facility_name, na=False)]

        county = county_dict[f'{facility_name}']
        try:
            county_facility_x_correlation(facility, nyt_data[nyt_data['county'].str.contains(county, na=False)], start_date, end_date,
                                          facility_name)
        except:
            continue


## Exploratory analysis for the local data for San Diego county scraped from the county website
def local_case_data(scrape_new=False):
    if scrape_new:
        get_sandiego_api_covid.get_juris()
    # with open('sandiego_all_data_response.json') as f:
    today = datetime.date.today()
    with open(f'sandiego_juris_data_scraped_{today.month}-{today.day}-{today.year}.json') as f:
        sandiego_json = json.load(f)
    sandiego_df = json_normalize(sandiego_json['features'])
    sandiego_df.head()
    with open('sandiego_zip_nov_data.json') as f:
        sandiego_zips_json = json.load(f)
    sandiego_zips_df = json_normalize(sandiego_zips_json['features'])
    chula_vista = sandiego_df[sandiego_df['attributes.name'].str.contains('CHULA VISTA')]
    rancho_sd = sandiego_df[sandiego_df['attributes.name'].str.contains('Rancho San Diego')]

    # TODO: once we have the date range, automate an API call for each range of dates for the zip data, to get that full df too
    # More next steps: organize all of this code
    # Plan: #identify nearby jurisdictions (and nerby zipcodes, that can be part 2) around Otay Mesa
    # Get time series for each jurisdiction of cases
    # Normalise cases by pop and add together / total pop, do time series correlation of Otay mesa data with surrounding area
    # Surrounding Otay Mesa zips and jurisdictions:
    # 91915, 91917, 91935, 91905 & 91906 & 91962 & 91963 & 91980 & 91934
    # 91901, 92019, 91914, 92021, 91977, 91978, 92154, 91913, 91911, 91910,
    # Rancho San Diego, Spring Valley, Alpine, La Presa, Chula Vista, National City

    # For each of the date ranges in the jurisdiction dataset, call this function and save the zip data:
    unique_date_ranges = sandiego_df['attributes.Update_Range'].unique()
    for item in unique_date_ranges:
        # NOTE: ImPORTANT!!
        # They switched to 03 for March (instead of just '3')
        if item is not None:
            first_date = datetime.datetime.strptime(item.split('-')[0],
                                                    '%m/%d/%Y')
            first_date_as_exact_str = item.split('-')[0].split('/')
            second_date = datetime.datetime.strptime(item.split('-')[1],
                                                     '%m/%d/%Y')
            second_date_as_exact_str = item.split('-')[1].split('/')

            if scrape_new:
                get_sandiego_api_covid.get_zip_by_date_range(first_date_as_exact_str[0], first_date_as_exact_str[1],
                                                             first_date_as_exact_str[2],
                                                             second_date_as_exact_str[0], second_date_as_exact_str[1],
                                                             second_date_as_exact_str[2])

            sandiego_zips_data_feb = get_sandiego_api_covid.read_json_file_for_date_range(first_date_as_exact_str[0],
                                                                                          first_date_as_exact_str[1],
                                                                                          first_date_as_exact_str[2],
                                                                                          second_date_as_exact_str[0],
                                                                                          second_date_as_exact_str[1],
                                                                                          second_date_as_exact_str[2])
            sandiego_zips_data_feb.head(5)

    ucla_Data = load_ucla_data()
    ice_facilities = select_ice_facilities(ucla_Data)
    ca_facilities = select_california_ice(ice_facilities)
    otay_mesa = ca_facilities[
        ca_facilities['Name'].str.contains('ICE OTAY MESA DETENTION CENTER SAN DIEGO CDF', na=False)]

    # Sample cross correlation with jurisidictional case counts
    unique_names = np.unique(sandiego_df[['attributes.name']])
    for name in unique_names:
        try:
            juris_covid_df = sandiego_df[sandiego_df['attributes.name'].str.contains(name)]
            cross_correlation_plot(juris_covid_df, otay_mesa)
        except:
            continue

    # Sample cross correlation with local data
    cross_correlation_plot(chula_vista, otay_mesa)
    cross_correlation_plot(rancho_sd, otay_mesa)

    return sandiego_df

## Cross correlation for 7 day cases average with local region (from the scraped San Diego data) and facility
def cross_correlation_plot(region, facility):
    region_name = region.head(1)['attributes.name']
    date_series_7_day = []
    date_series_7_day_labels = []
    good_data_idx = []
    for i in range(len(region)):
        try:
            first_date = datetime.datetime.strptime(region.iloc[i]['attributes.Update_Range'].split('-')[0],
                                                    '%m/%d/%Y')
            second_date = datetime.datetime.strptime(region.iloc[i]['attributes.Update_Range'].split('-')[1],
                                                     '%m/%d/%Y')
            print(first_date, second_date)
            date_series_7_day.append(second_date)
            date_series_7_day_labels.append(f'{first_date} - {second_date}')
            good_data_idx.append(i)
        except AttributeError:
            pesky_date = region.iloc[i]['attributes.Update_Range']
            print(f'Couldn not parse date {pesky_date}')
    plt.plot(date_series_7_day, region['attributes.Cases_7_Days'].iloc[good_data_idx])
    plt.xticks(ticks=date_series_7_day, rotation=30)
    plt.title(f'{region_name} - Cases 7 Day Avg')
    plt.show()

    start_date = date_series_7_day[0]
    end_date = date_series_7_day[-1]

    facility['Date'] = pd.to_datetime(facility['Date'])
    facility_mask = (facility['Date'] > start_date) & (facility['Date'] <= end_date)
    facility = facility.loc[facility_mask]

    chunk_val = int(np.floor(len(facility) / len(date_series_7_day)))

    d1 = pd.Series(facility['Residents.Confirmed'].rolling(7).obj.iloc[::chunk_val].values)
    facility['Rolling'] = d1
    d2 = pd.Series(region['attributes.Cases_7_Days'].iloc[good_data_idx].rolling(7).obj.values)
    seconds = 5
    fps = 30

    cx_0 = crosscorr(d1, d2, 0)
    rs = np.array([crosscorr(d1, d2, lag) for lag in range(-len(d1), len(d1))])
    rs_not_nan = rs[~ np.isnan(rs)]
    offset = np.floor(len(rs) / 2) - np.argmax(rs_not_nan)
    f, ax = plt.subplots(figsize=(14, 5))
    ax.plot(rs)
    ax.axvline(np.ceil(len(rs) / 2), color='k', linestyle='--', label='Center')
    ax.axvline(np.argmax(rs_not_nan), color='r', linestyle='--', label='Peak synchrony')
    ax.set(title=f'OTAY MESA cross correlation with {region_name} juris \n Date Offset = {offset} frames',
           xlabel='Offset',
           ylabel='Pearson r')
    # ax.set_xticks(np.arange(len(joined_df)))
    # ax.set_xticklabels(joined_df['Date'])
    plt.legend()
    plt.show()

def moving_average(a, n=7):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def new_outbreak_correlation():
    # TODO w Minali:
    # Can we incorporate into here the code Neal wrote for computing active cases, and then do cross correlation with that
    # Residents.Active column is there now! So can use that for correlation
    # Argument could be, facilities arent exempt from county increasing rates, all the more push for vaccination efforts
    #
    facility_names = ['ICE ADAMS COUNTY DETENTION CENTER', 'ICE KARNES COUNTY RESIDENTIAL CENTER', 'ICE SOUTH TEXAS DETENTION COMPLEX',
                      'ICE SOUTH TEXAS FAMILY RESIDENTIAL CENTER', 'ICE WINN CORRECTIONAL CENTER']
    county_dict = {'ICE ADAMS COUNTY DETENTION CENTER': 'Adams', 'ICE KARNES COUNTY RESIDENTIAL CENTER': 'Karnes', 'ICE SOUTH TEXAS DETENTION COMPLEX':'Frio',
                   'ICE SOUTH TEXAS FAMILY RESIDENTIAL CENTER':'Frio', 'ICE WINN CORRECTIONAL CENTER':'Winn'}
    state_dict = {'ICE ADAMS COUNTY DETENTION CENTER' : 'Mississippi', 'ICE KARNES COUNTY RESIDENTIAL CENTER':'Texas',
                  'ICE SOUTH TEXAS DETENTION COMPLEX':'Texas', 'ICE SOUTH TEXAS FAMILY RESIDENTIAL CENTER':'Texas',
                  'ICE WINN CORRECTIONAL CENTER':'Louisiana'}
    pop_dict = {'ICE ADAMS COUNTY DETENTION CENTER': 30693, 'ICE KARNES COUNTY RESIDENTIAL CENTER': 15545, 'ICE SOUTH TEXAS DETENTION COMPLEX':20306,
                   'ICE SOUTH TEXAS FAMILY RESIDENTIAL CENTER':20326, 'ICE WINN CORRECTIONAL CENTER': 14313}
    ice_facilities = select_ice_facilities(load_ucla_data())
    covid_df = load_nyt_data()
    for facility in facility_names:
        df = ice_facilities[ice_facilities['Name'].str.contains(f'{facility}', na=False)]
        county = covid_df[covid_df['county'] == county_dict[facility]]
        county = county[county['state'] == state_dict[facility]]
        county_pop = pop_dict[facility]
        county_facility_x_correlation(df, county, datetime.datetime.strptime('02-01-2021', '%m-%d-%Y'),
                                      datetime.datetime.strptime('04-12-2021', '%m-%d-%Y'), facility, county_pop)

def model_with_data():
    base_color = '#555526'
    scatter_color = '#92926C'
    more_colors = [ "#D7790F", "#82CAA4", "#4C6788", "#84816F",
                "#71A9C9", "#AE91A8"]
    ## Combination disease model with data from datatools
    # Can observe c_jail from real data, comparing the per-10000 person rate. Then show a model which makes for a compelling argument
    # that even if spread is dying down in the community, it can still have an affect on within-detention case rates.
    # example_model(county_pop=30693, c_jail=160, staff_pop=50, detention_pop=338, init_community_infections=2200, show_recovered=True)
    # ICE SOUTH TEXAS DETENTION COMPLEX: params inferred from data, avg c_jail, should get to 2700 infections after 60 days
    # initial detention infections is 300, should get to 400 to match reports
    plt.figure('model')
    # model_result = model.example_model(county_pop=20306, c_jail=17, staff_pop=100, detention_pop=650, init_community_infections=2403, init_detention_infections=300, show_recovered=True)
    # model.example_model(county_pop=20306, staff_pop=200, detention_pop=650, show_recovered=False, show_susceptible=False,
    #               beta=2.43,sigma=0.5, gamma=1 / 10, gamma_ei=1 / 6.7, staff_work_shift=3, c_jail=3, c_0=500,
    #               init_community_infections=244, init_detention_infections=30,arrest_rate=.0001, alos=.033,
    #               normalize=True, same_plot=True, num_days=80)
    # the above params end up callibrating to: Beta community/detention Params: 0.059250950941187944, 4.288235294117648

    facility_names = ['ICE ADAMS COUNTY DETENTION CENTER', 'ICE KARNES COUNTY RESIDENTIAL CENTER', 'ICE SOUTH TEXAS DETENTION COMPLEX',
                      'ICE SOUTH TEXAS FAMILY RESIDENTIAL CENTER', 'ICE WINN CORRECTIONAL CENTER']
    county_dict = {'ICE ADAMS COUNTY DETENTION CENTER': 'Adams', 'ICE KARNES COUNTY RESIDENTIAL CENTER': 'Karnes', 'ICE SOUTH TEXAS DETENTION COMPLEX':'Frio',
                   'ICE SOUTH TEXAS FAMILY RESIDENTIAL CENTER':'Frio', 'ICE WINN CORRECTIONAL CENTER':'Winn'}
    state_dict = {'ICE ADAMS COUNTY DETENTION CENTER' : 'Mississippi', 'ICE KARNES COUNTY RESIDENTIAL CENTER':'Texas',
                  'ICE SOUTH TEXAS DETENTION COMPLEX':'Texas', 'ICE SOUTH TEXAS FAMILY RESIDENTIAL CENTER':'Texas',
                  'ICE WINN CORRECTIONAL CENTER':'Louisiana'}
    pop_dict = {'ICE ADAMS COUNTY DETENTION CENTER': 30693, 'ICE KARNES COUNTY RESIDENTIAL CENTER': 15545, 'ICE SOUTH TEXAS DETENTION COMPLEX':20306,
                   'ICE SOUTH TEXAS FAMILY RESIDENTIAL CENTER':20326, 'ICE WINN CORRECTIONAL CENTER': 14313}
    # Parameter set dictionary for each facility we are looking at, so we can plug right into the model
    param_dict = {'ICE ADAMS COUNTY DETENTION CENTER': model.ModelParams(county_pop=30693, staff_pop=50, #a guess
                                                                        detention_pop=338, beta=2.43,sigma=0.5,
                                                                        gamma=1 / 10, gamma_ei=1 / 6.7,
                                                                        staff_work_shift=3, c_jail=1, c_0=750,
                                                                        init_community_infections=35,
                                                                        init_detention_infections=3,arrest_rate=.0001,
                                                                        alos=.033),
                  'ICE KARNES COUNTY RESIDENTIAL CENTER': None,
                  'ICE SOUTH TEXAS DETENTION COMPLEX': model.ModelParams(county_pop=20306, staff_pop=200,
                                                                        detention_pop=650, beta=2.43,sigma=0.5,
                                                                        gamma=1 / 10, gamma_ei=1 / 6.7,
                                                                        staff_work_shift=3, c_jail=3, c_0=500,
                                                                        init_community_infections=244,
                                                                        init_detention_infections=30,arrest_rate=.0001,
                                                                        alos=.033),
                   'ICE SOUTH TEXAS FAMILY RESIDENTIAL CENTER':None, 'ICE WINN CORRECTIONAL CENTER': None}
    y_lim_dict = {'ICE ADAMS COUNTY DETENTION CENTER': [[10**(-3), 10**(-2), 10**(-1)], ['10', '100', '1,000'], 10**(-3), 10**(-1)],
                  'ICE SOUTH TEXAS DETENTION COMPLEX': [[10**(-3), 10**(-2), 10**(-1)], ['10', '100', '1,000'], 10**(-3)-.0001, 10**(-1)+.1]}

    ice_facilities = select_ice_facilities(load_ucla_data())
    covid_df = load_nyt_data()
    facility_name = 'ICE SOUTH TEXAS DETENTION COMPLEX' #Do this for each one with rising cases
    # facility_name = 'ICE ADAMS COUNTY DETENTION CENTER'
    model.example_model(model_params=param_dict[facility_name], num_days=80, normalize=True, same_plot=True,
                        show_recovered=False, show_susceptible=False)
    df = ice_facilities[ice_facilities['Name'].str.contains(f'{facility_name}', na=False)]
    county = covid_df[covid_df['county'] == county_dict[facility_name]]
    county = county[county['state'] == state_dict[facility_name]]
    county_pop = pop_dict[facility_name]

    county_name = county.head(1)['county'].values[0]
    start_date = datetime.datetime.strptime('02-01-2021', '%m-%d-%Y')
    start_date = pd.to_datetime(start_date)
    end_date = datetime.datetime.strptime('05-01-2021', '%m-%d-%Y')
    end_date = pd.to_datetime(end_date)
    df['Date'] = pd.to_datetime(df['Date'])
    facility_mask = (df['Date'] > start_date) & (df['Date'] <= end_date)
    facility = df.loc[facility_mask]

    county['date'] = pd.to_datetime(county['date'])
    county_mask = (county['date'] > start_date) & (county['date'] <= end_date)
    county = county.loc[county_mask]

    joined_df = county.join(facility.set_index('Date'), on='date', how='left')
    # TODO the thing that would make this make more sense as well is to change the ODE's to bring in new population over time
    # say 3 people per day, and 3 out, to general pop
    plt.figure('compare case rates')
    avg_difference_in_rates = np.average(joined_df['Residents.Active'].fillna(method='ffill')[1:] / facility.head(1)['Population.Feb20'].values[0] * 10000)/np.average(joined_df['cases'].diff(10).fillna(method='bfill') / county_pop * 10000)
    plt.ylabel('Active case rate per 10,000 people')
    plt.title(f'Active case rates for {facility_name} and surrounding county\n'
              f'Avg rate of detention facility is {np.round(avg_difference_in_rates,1)}X higher than county rate')
    plt.plot(joined_df['date'],
             joined_df['Residents.Active'].fillna(method='ffill') / facility.head(1)['Population.Feb20'].values[0],
             label=f'{facility_name} Detainee Rate', color=more_colors[0])
    plt.plot(joined_df['date'], joined_df['cases'].diff(10).fillna(method='bfill') / county_pop,
             label='County rate', color=more_colors[1])
    plt.xticks(rotation=35)
    # plt.ylim(1, 100000)
    # plt.semilogy()
    # plt.yticks([10, 100, 1000, 10000], labels=['10', '100', '1000', '10000'])

    # plt.show()

    plt.figure('model')
    # TODO: need to fix the y axes labels with percentage vs proportion vs log etc.
    plt.scatter(np.arange(82), moving_average(np.array(joined_df['Residents.Active'].fillna(method='ffill')[1:])) / facility.head(1)['Population.Feb20'].values[0],
                label='Active cases Detainees', s=1, color=scatter_color)
    plt.scatter(np.arange(83), moving_average(np.array(joined_df['cases'].diff(10).fillna(method='bfill'))) / county_pop,
                label='Estimated active cases County', s=1, color=base_color)
    plt.legend(loc='upper left')
    # plt.ylim([0, 0.15])
    plt.semilogy()
    plt.yticks(y_lim_dict[facility_name][0], y_lim_dict[facility_name][1])
    plt.ylim([y_lim_dict[facility_name][2], y_lim_dict[facility_name][3]])
    plt.xlabel(f'Days past {start_date}')
    # plt.xticks(np.arange(0, 80, 10), joined_df['date'][0, 10, 20, 30, 40, 50, 60, 70])
    # plt.ylim([.0005, .15])
    print(joined_df['Residents.Active'].head(5))
    plt.show()


if __name__ == '__main__':
    ### Example:
    # local_case_data(scrape_new=False)
    # ca_ice_cross_correlation()
    model_with_data()
    # new_outbreak_correlation()

## Notes
## To do: for Exploratory section:
## Function that plots the auto-correlation between a date range as selected between a county and an ice facility
## autocorrelation with local hospitals
## Next: do rolling 7-day average smoothed
## Incidence of a 10-day lag
## Try to autocorrelate incidence with each other?