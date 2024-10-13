import xarray as xr
import numpy as np
import os 
import datetime as dt
import rioxarray
from geometry import region_names, plot_region, shape_file, get_region_shape
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import mapping
from collections import defaultdict
import time
import pickle
import json 

def flooded_pixel_counts(df, clip=False, region=None):
    # if a clipping region is given, clip
    if clip == True:
        df = df.rio.write_crs(region.crs)
        try:
            df = df.rio.clip(region.geometry.values, region.crs)
        except:
            return [0,0,0]
        water_detection = df['WaterDetection']
        water_detection = water_detection.sel(band=1)
    else:
        water_detection = df['WaterDetection']

    water_detection_data = water_detection.values
    unique_values, counts = np.unique(water_detection_data, return_counts=True)

    # Create a dictionary to map unique values to their counts
    value_counts = dict(zip(unique_values, counts))

    #calculate percentage of pixels with some flooding
    total_pixels = water_detection.size
    mask = (water_detection >= 100) & (water_detection <= 200)
    flooded_pixel_count = mask.sum().item()

    # calculate adjusted flooded pixel percentage
    scaled_flooding = 0
    for value, count in value_counts.items():
        if (value >= 100) & (value <= 200):
            scaled_flooding += ((value - 100) / 100) * count

    return  total_pixels, flooded_pixel_count, scaled_flooding

def dd():
    return defaultdict(ddd)

def ddd():
    return np.zeros(3, dtype=np.int64)

def process_data():
    print('yes')
    regions = region_names('adm2')
    regions_shapes = {name: get_region_shape(name) for name in regions}

    root = os.path.join('/Users/aws/Documents/personal_projects/african_parks/data/viirs')

    for year in sorted(os.listdir(root)):

        # Initialize the nested defaultdict with numpy arrays
        all_data = defaultdict(dd)

        year_dir = os.path.join(root, year)
        if year.startswith('.'):
            continue  # Skip hidden directories (e.g., .DS_Store)
        
        for month in sorted(os.listdir(year_dir)):
            print(year, month)
            month_dir = os.path.join(year_dir, month)
            if month.startswith('.'):
                continue  # Skip hidden directories (e.g., .DS_Store)
            
            for day in sorted(os.listdir(month_dir)):
                day_dir = os.path.join(month_dir, day)
                if day.startswith('.'):
                    continue  # Skip hidden directories (e.g., .DS_Store)
                
                for file in os.listdir(day_dir):
                    if file == '.DS_Store':
                        continue  # Skip .DS_Store files
                    
                    file = os.path.join(day_dir, file)
                    ds = rioxarray.open_rasterio(file)

                    for region in regions_shapes.keys():
                        t, f, fa = flooded_pixel_counts(ds, clip=True, region=regions_shapes[region])
                        all_data[region][f'{year}_{month}_{day}'] += np.array([t,f,fa], dtype=np.int64)
        
        # Ensure 'data_reduced' directory exists
        output_dir = 'data_reduced'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save the data for the year
        with open(f'{output_dir}/{year}.pickle', 'wb') as f:
            pickle.dump(all_data, f)

process_data()

# Function to load a pickle file
def load_pickle(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)
    
def pickle_to_xarray(pickle_dir):

    # Directory containing the pickle files
    directory = pickle_dir

    # Load all pickle files into a list of dictionaries
    data_list = [load_pickle(os.path.join(directory, file)) for file in os.listdir(directory)]

    # Aggregate unique dates and regions
    dates = sorted({date for data in data_list for region in data.values() for date in region})
    regions = sorted({region for data in data_list for region in data})
    values = ['total_pixels', 'flooded_pixels', 'adjusted_flooded_pixels']

    # Initialize the data structure for xarray
    transformed_data = {value: (['date', 'region'], np.full((len(dates), len(regions)), np.nan)) for value in values}

    # Fill the arrays with data from all dictionaries
    for data in data_list:
        for region, date_values in data.items():
            for date, values_list in date_values.items():
                date_idx = dates.index(date)
                region_idx = regions.index(region)
                for value_idx, value in enumerate(values):
                    transformed_data[value][1][date_idx, region_idx] = values_list[value_idx]

    # Create the xarray.Dataset
    dataset = xr.Dataset(
        transformed_data,
        coords={
            'date': dates,
            'region': regions
        }
    )

    save_dir = os.path.join(os.getcwd(), 'final_data_reduced/final_data.nc')
    dataset.to_netcdf(save_dir)

#pickle_to_xarray(os.path.join(os.getcwd(), 'data_reduced'))