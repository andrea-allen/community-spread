from coviddata import datatools

if __name__ == '__main__':
    print('UCLA Covid Behind Bars Data Project')
    datatools.load_ucla_data()
    datatools.facility_density_map()