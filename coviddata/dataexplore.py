
import pandas as pd

def load_data():
    data = pd.read_csv("http://104.131.72.50:3838/scraper_data/summary_data/scraped_time_series.csv")
    return data