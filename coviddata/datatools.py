import pandas as pd
import matplotlib.pyplot as plt
import folium
import numpy as np
from pandas.io.json import json_normalize
import json
import datetime
from coviddata import get_sandiego_api_covid


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


def county_facility_x_correlation(facility, county, start_date, end_date, facility_name):
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

    plt.plot(facility['Date'], facility['Residents.Confirmed'].rolling(7).obj, color='blue')
    plt.xticks(rotation=45)
    plt.xlabel('Days')
    plt.ylabel('Cumulative case count')
    plt.title(f'7 Day Rolling Avg - {facility_name}')
    plt.show()

    plt.plot(county['date'], county['cases'].rolling(7).obj, color='orange')
    plt.xticks(rotation=45)
    plt.xlabel('Days')
    plt.ylabel('Cumulative case count')
    plt.title(f'7 Day Rolling Avg - County {county_name}')
    plt.show()

    joined_df = facility.join(county.set_index('date'), on='Date', how='left')

    ## TODO before doing the correlation, need to join on the date column to get the same date values for NYT and ICE data
    ## basically need to build up some more of my data tools first
    # Compute rolling window synchrony
    d1 = joined_df['Residents.Confirmed'].rolling(7).obj
    d2 = joined_df['cases'].rolling(7).obj
    rs = np.array([crosscorr(d1, d2, lag) for lag in range(-len(joined_df), len(joined_df))])
    rs_not_nan = rs[~ np.isnan(rs)]
    offset = np.floor(len(rs) / 2) - np.argmax(rs_not_nan)
    f, ax = plt.subplots(figsize=(14, 5))
    ax.plot(rs)
    ax.axvline(np.ceil(len(rs) / 2), color='k', linestyle='--', label='Center')
    ax.axvline(np.argmax(rs_not_nan), color='r', linestyle='--', label='Peak synchrony')
    ax.set(title=f'{facility_name} cross correlation with {county_name} county \n Date Offset = {offset} frames',
           xlabel='Offset',
           ylabel='Pearson r')
    # ax.set_xticks(np.arange(len(joined_df)))
    # ax.set_xticklabels(joined_df['Date'])
    plt.legend()
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


if __name__ == '__main__':
    ### Example:
    local_case_data(scrape_new=False)
    ca_ice_cross_correlation()

## Notes
## To do: for Exploratory section:
## Function that plots the auto-correlation between a date range as selected between a county and an ice facility
## autocorrelation with local hospitals
## Next: do rolling 7-day average smoothed
## Incidence of a 10-day lag
## Try to autocorrelate incidence with each other?