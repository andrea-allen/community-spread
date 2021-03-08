
import pandas as pd
import matplotlib.pyplot as plt
import folium

def load_nyt_data():
    nyt_url_counties = 'https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv'
    df = pd.read_csv(nyt_url_counties, na_filter=True)
    return df

def load_ucla_data():
    data = pd.read_csv("http://104.131.72.50:3838/scraper_data/summary_data/scraped_time_series.csv")
    return data

def select_ice_facilities(data):
    ice_data = data[data['Jurisdiction'].str.contains('immigration', na=False)]
    print(ice_data.head(10))
    return ice_data

def facility_density_map():
    ice_data = select_ice_facilities(load_ucla_data())
    m = folium.Map(location=[41, -97], zoom_start=5)
    us_counties = 'https://gist.githubusercontent.com/wrobstory/5586482/raw/6031540596a4ff6cbfee13a5fc894588422fd3e6/us-counties.json'
    folium.Choropleth(
        geo_data=us_counties,
        name='choropleth',
        data=ice_data,
        columns=['Facility.ID', 'Residents.Confirmed'],
        key_on='feature.id',
        fill_color='Blues',
        fill_opacity=0.75,
        nan_fill_opacity=0.0,
        line_opacity=1
    ).add_to(m)
    plt.plot(m)
    plt.show()

def case_density_map():
    # covid_data = load_nyt_data()
    ucla_data = pd.read_csv("http://104.131.72.50:3838/scraper_data/summary_data/scraped_time_series.csv")
    county_name_to_fips = pd.read_csv("https://raw.githubusercontent.com/kjhealy/fips-codes/master/state_and_county_fips_master.csv", header=0, delimiter=',')
    state_name_to_abbrv = pd.read_csv("coviddata/name-abbr.csv")
    state_name_to_abbrv['NAME'] = state_name_to_abbrv['NAME'].str.lower()
    county_name_to_fips['name'] = county_name_to_fips['name'].str.lower()
    # Goal:
    # County Name -- Number of Ice facilities -- Number of jails or prisons

    # How to get there
    # Group by facility ID and get the county value for each first row in each group
    facility_ids = ucla_data.groupby('Facility.ID').first()
    facility_ids['County'] = facility_ids['County'].str.lower()
    facility_ids['State'] = facility_ids['State'].str.lower()
    cleaned_df = pd.DataFrame(facility_ids[['State', 'County', 'Jurisdiction']])
    cleaned_df = cleaned_df.join(state_name_to_abbrv.set_index('NAME'), on='State')
    # TODO pick up here, map county to string containing
    cleaned_df = cleaned_df.join(county_name_to_fips.set_index('name'), on=['ABBRV', 'County'])
    other_facilities = cleaned_df[~cleaned_df['Jurisdiction'].str.contains('immigration')]
    ice_facilities = cleaned_df[cleaned_df['Jurisdiction'].str.contains('immigration', na=False)]
    other_facs_per_county = other_facilities.groupby(['State', 'County']).count()
    ice_facs_per_county = ice_facilities.groupby(['State', 'County']).count()
    # pipe that into Facility ID -- StateCountyComb name -- Jurisdtiction (Y/N Ice)
    by_county = facility_ids.groupby('County')
    # Then split off by jurisdiction
    # Group by county name and get count, to get county name -- number ICE -- number other
    # Plot
    m = folium.Map(location=[41, -97], zoom_start=5)
    us_counties = 'https://gist.githubusercontent.com/wrobstory/5586482/raw/6031540596a4ff6cbfee13a5fc894588422fd3e6/us-counties.json'
    folium.Choropleth(
        geo_data=us_counties,
        name='choropleth',
        data=covid_data,
        columns=['fips', 'cases'],
        key_on='feature.id',
        fill_color='Blues',
        fill_opacity=0.75,
        nan_fill_opacity=0.0,
        line_opacity=1
    ).add_to(m)
    m.save('index.html')
    # plt.plot(m)
    # plt.show()