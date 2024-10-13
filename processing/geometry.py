import geopandas as gpd
import os
import matplotlib.pyplot as plt

def get_region_shape(name):
    # Check if the name exists in either 'ADM1_NAME' or 'ADM2_NAME' column
    path = os.path.join('/Users/aws/Documents/personal_projects/african_parks/regional_shape_files/example.shp')
    gdf = gpd.read_file(path)
    if not gdf['ADM1_NAME'].isin([name]).any() and not gdf['ADM2_NAME'].isin([name]).any():
        print(f'Region with name "{name}" not found.')
        return
    region = gdf[(gdf['ADM1_NAME'] == name) | (gdf['ADM2_NAME'] == name)]
    return region

def shape_file():
    path = os.path.join('/Users/aws/Documents/personal_projects/african_parks/regional_shape_files/example.shp')
    gdf = gpd.read_file(path)
    return gdf

def region_names(level=None):
    path = os.path.join('/Users/aws/Documents/personal_projects/african_parks/regional_shape_files/example.shp')
    gdf = gpd.read_file(path)
    
    if level == 'adm1':
        return gdf['ADM1_NAME'].tolist()
    
    elif level == 'adm2':
        return gdf['ADM2_NAME'].tolist()

    names = {'adm1': gdf['ADM1_NAME'].tolist(), 'adm2': gdf['ADM2_NAME'].tolist()}
    return names

def plot_region(name):
    # Path to your shapefile
    path = os.path.join('/Users/aws/Documents/personal_projects/african_parks/regional_shape_files/example.shp')
    
    # Read the shapefile into a GeoDataFrame
    gdf = gpd.read_file(path)
    
    # Check if the name exists in either 'ADM1_NAME' or 'ADM2_NAME' column
    if not gdf['ADM1_NAME'].isin([name]).any() and not gdf['ADM2_NAME'].isin([name]).any():
        print(f'Region with name "{name}" not found.')
        return
    
    # Plot the shapefile without the red outline first
    gdf.plot()

    # Filter GeoDataFrame to only include the specific region
    region = gdf[(gdf['ADM1_NAME'] == name) | (gdf['ADM2_NAME'] == name)]

    # Plot the region with red outline
    region.plot(edgecolor='red', linewidth=2, ax=plt.gca())
    
    # Customize the plot (optional)
    plt.title(f'Region: {name}')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True)

    # Show the plot
    plt.show()