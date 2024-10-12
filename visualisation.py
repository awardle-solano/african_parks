import xarray as xr
import numpy as np
import plotly.graph_objs as go
import panel as pn
import rioxarray 
import xarray
import os
import datetime
from geometry import region_names, plot_region, shape_file, get_region_shape
import matplotlib.pyplot as plt
import imageio
import pandas as pd

file_path = '/Users/aws/Documents/personal_projects/african_parks/final_data_reduced/final_data.nc'

dataset = xr.open_dataset(file_path)

selected_regions = dataset['region'].values.tolist()

# Function to create the plot
def create_plot(dataset, selected_regions):
    fig = go.Figure()
    for region in selected_regions:
        ratio = dataset['adjusted_flooded_pixels'].sel(region=region) / dataset['total_pixels'].sel(region=region)
        fig.add_trace(go.Scatter(x=dataset['date'].values, y=ratio.values, mode='lines', name=region))
    
    fig.update_layout(
        title="Interactive Plot of %Flooding by Region",
        xaxis_title="Date",
        yaxis_title="%Flooded Pixels",
        template="plotly_white"
    )
    return fig

# Create a list of checkboxes for regions
region_checkboxes = pn.widgets.CheckBoxGroup(name='Regions', options=selected_regions, value=selected_regions, inline=True)

# Create a panel for the plot
@pn.depends(region_checkboxes.param.value)
def update_plot(selected_regions):
    return create_plot(dataset, selected_regions)

# Layout
layout = pn.Column(
    region_checkboxes,
    update_plot
)

html_output = layout.save('interactive_plot.html', embed=True)

# Serve the app
#pn.serve(layout)


'''
regions = region_names('adm2')
regions_shapes = {name: get_region_shape(name) for name in regions}

root = os.path.join(os.getcwd(), 'data')

for year in sorted(os.listdir(root)):

    year_dir = os.path.join(root, year)
    if year.startswith('.'):
        continue  # Skip hidden directories (e.g., .DS_Store)

    images = []  # List to store images for GIF creation
    
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
                elif '97' in file:
                    continue
                
                file = os.path.join(day_dir, file)
                ds = rioxarray.open_rasterio(file)
                
                region = regions_shapes['Nahr Lol']
                ds = ds.rio.write_crs(region.crs)
                
                ds = ds.rio.clip(region.geometry.values, region.crs)['WaterDetection'].sel(band=1)
                
                flood_mask = (ds >= 100) & (ds <= 200)

                time = [pd.to_datetime(f'{year}-{month}-{day}')]
                plt.figure()
                plt.contourf(ds['x'], ds['y'], ds.values, levels=np.linspace(100, 200, 101), cmap='viridis')
                plt.text(0.95, 0.05, f'{year}-{month}-{day}', horizontalalignment='right',
                    verticalalignment='bottom', transform=plt.gca().transAxes, fontsize=12, color='white',
                    bbox=dict(facecolor='black', alpha=0.5))
                for geom in region.geometry:
                    x, y = geom.exterior.xy
                    plt.plot(x, y, color='red', linewidth=2)
                plt.xlabel('Longitude')  # Add x-axis title
                plt.ylabel('Latitude')   # Add y-axis title
                
                # Save the plot with a date-based filename
                output_dir = os.getcwd()  # Use the current working directory or specify another directory
                output_path = os.path.join(output_dir, 'gif_data', f'{year}_{month}_{day}.png')  # Ensure month and day are zero-padded

                # Ensure the directory exists
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                plt.savefig(output_path)
                plt.close()
                images.append(output_path) 

    gif_path = os.path.join(output_dir, f'{year}.gif')
    with imageio.get_writer(gif_path, mode='I') as writer:
        for image in images:
            frame = imageio.imread(image)
            writer.append_data(frame)

    '''