
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
    ucla_data = pd.read_csv("http://104.131.72.50:3838/scraper_data/summary_data/scraped_time_series.csv")
    # Getting county data
    county_name_to_fips = pd.read_csv("https://raw.githubusercontent.com/kjhealy/fips-codes/master/state_and_county_fips_master.csv", header=0, delimiter=',')
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
    cleaned_df = cleaned_df.join(county_name_to_fips.set_index(['state','name']), on=[' ABBRV', 'county_appended'])

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