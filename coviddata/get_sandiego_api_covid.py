import requests
import json
from pandas.io.json import json_normalize
import datetime

def get_juris():
    url = "https://services1.arcgis.com/1vIhDJwtG5eNmiqX/arcgis/rest/services/CityJurisPts_CaseRate_PUBLIC_VIEW/FeatureServer/1/query?f=json&returnGeometry=false&spatialRel=esriSpatialRelIntersects&geometry=%7B%22xmin%22%3A-13089593.880859569%2C%22ymin%22%3A3794329.869664724%2C%22xmax%22%3A-12914705.960143171%2C%22ymax%22%3A3961268.3394394685%2C%22spatialReference%22%3A%7B%22wkid%22%3A102100%7D%7D&geometryType=esriGeometryEnvelope&inSR=102100&outFields=*&orderByFields=name%20asc&outSR=102100&resultOffset=0&resultRecordCount=32000&resultType=standard&cacheHint=false"

    payload = {}
    headers = {}

    response = requests.request("GET", url, headers=headers, data=payload)

    print(response.text)

    today = datetime.date.today()
    with open(f'sandiego_juris_data_scraped_{today.month}-{today.day}-{today.year}.json', 'w') as f:
        json.dump(response.json(), f, indent=4)

    return response.json()

def get_zip_by_date_range(m1, d1, y1, m2, d2, y2):
    url = "https://services1.arcgis.com/1vIhDJwtG5eNmiqX/arcgis/rest/services/CaseRateZIPs_MainDash_PUBLIC_VIEW" \
          f"/FeatureServer/0/query?f=json&where=(Update_Range%20%3D%20%27{m1}%2F{d1}%2F{y1}-{m2}%2F{d2}%2F{y2}%27)%20AND%20(" \
          "episode7day%20IS%20NOT%20NULL)&returnGeometry=false&spatialRel=esriSpatialRelIntersects&outFields=*&outSR" \
          "=102100&resultOffset=0&resultRecordCount=50&resultType=standard&cacheHint=true "

    payload = {}
    headers = {}

    response = requests.request("GET", url, headers=headers, data=payload)

    print(response.text)

    with open(f'sandiego_zip_data{m1}-{d1}-{y1}_{m2}-{d2}-{y2}.json', 'w') as f:
        json.dump(response.json(), f, indent=4)
    return response.json()

def read_json_file_for_date_range(m1, d1, y1, m2, d2, y2):
    with open(f'sandiego_zip_data{m1}-{d1}-{y1}_{m2}-{d2}-{y2}.json') as f:
        sandiego_zips_json = json.load(f)
    sandiego_zips_df = json_normalize(sandiego_zips_json['features'])
    return sandiego_zips_df
