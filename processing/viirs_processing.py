import ee
import geemap
import geopandas as gpd
import rioxarray as rxr
import requests
from datetime import datetime, timedelta
from tqdm import tqdm

# Authenticate and initialize Earth Engine
ee.Authenticate()
ee.Initialize()

# Load your GeoDataFrame (gdf) containing the geometry
gdf = gpd.read_file("/Users/aws/Documents/personal_projects/african_parks/regional_shape_files/example.shp")  # Replace with your shapefile or GeoDataFrame

# Convert the GeoDataFrame to an Earth Engine geometry
ee_geometry = geemap.geopandas_to_ee(gdf)

# Function to download image directly from Earth Engine
def download_image(image, filename, geometry):
    # Ensure that 'geometry' is a geometry object, not a FeatureCollection
    geometry = geometry.geometry()  # Get the geometry from FeatureCollection if necessary
    
    url = image.getDownloadURL({
        'scale': 5000,  # Adjust the scale (resolution)
        'region': geometry.bounds().getInfo()['coordinates'],  # Set region to geometry bounds
        'format': 'GeoTIFF'
    })
    
    # Download the image
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded {filename}")
    else:
        print(f"Failed to download {filename}")

# Function to process the data by date range
def process_by_date_range(start_date, end_date, geometry, region_name):
    # Load and filter the CHIRPS daily precipitation data for the given date range
    chirps_collection = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY") \
                         .filterDate(start_date, end_date) \
                         .filterBounds(geometry)

    # Download each image as GeoTIFF
    image_list = chirps_collection.toList(chirps_collection.size())
    for i in range(chirps_collection.size().getInfo()):
        img = ee.Image(image_list.get(i))

        # Get the acquisition date from the 'system:time_start' property
        img_date = img.date().format('YYYY-MM-dd').getInfo()

        # Use the actual image date in the filename
        filename = f'{region_name}_chirps_{img_date}.tif'  # Use actual date for filename
        download_image(img, filename, geometry)

# Function to divide the date range into chunks
def split_date_range(start_date, end_date, days_per_chunk=30):
    current_date = start_date
    while current_date < end_date:
        next_date = min(current_date + timedelta(days=days_per_chunk), end_date)
        yield current_date, next_date
        current_date = next_date

# Main process
start_date = datetime(2012, 1, 1)
end_date = datetime(2024, 7, 31)

# Process the data in chunks of 3 months (90 days)
for start, end in tqdm(split_date_range(start_date, end_date, days_per_chunk=90)):
    process_by_date_range(start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'), ee_geometry, "region")

print("All data downloaded and saved as NetCDF files.")


