from coviddata import dataexplore

if __name__ == '__main__':
    print('UCLA Covid Behind Bars Data Project')
    dataexplore.load_ucla_data()
    dataexplore.facility_density_map()