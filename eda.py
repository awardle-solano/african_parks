import geopandas as gpd
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import os
import imageio.v2 as imageio
from matplotlib.colors import TwoSlopeNorm
from scipy import stats

def create_flooding_gif(shapefile_dir, data_dir, output_gif='flooding_animation.gif', frame_duration=2):
    """
    Function to create an animated GIF showing flooding percentage per region over time.
    
    Parameters:
    - shapefile_dir: Path to the shapefile containing region geometries.
    - data_dir: Path to the xarray dataset containing flooding data.
    - output_gif: The name of the output GIF file (default: 'flooding_animation.gif').
    - frame_duration: Duration of each frame in the GIF in seconds (default: 2 seconds).
    """
    # Load geometries (GeoDataFrame)
    gdf = gpd.read_file(shapefile_dir)

    # Load the dataset (xarray Dataset)
    ds = xr.open_dataset(data_dir)

    # Convert the 'date' column to datetime type
    ds['date'] = pd.to_datetime(ds['date'].values, format='%Y_%m_%d')

    # Resample the data by month, taking the mean for each month
    ds_monthly = ds.resample(date="1ME").mean()
    ds_monthly['flood_percent'] = ds_monthly.flooded_pixels / ds_monthly.total_pixels

    # Calculate min and max flooding percentage to fix axes
    flood_percent_min = ds_monthly['flood_percent'].min()
    flood_percent_max = ds_monthly['flood_percent'].max()

    # Create output directory for the frames
    output_dir = 'frames'
    os.makedirs(output_dir, exist_ok=True)

    # Store file paths of generated frames
    frames = []

    # Iterate over each time step (assuming 'date' is the time dimension)
    for time_step in ds_monthly.date:
        print(time_step)
        # Select the data for the current time step
        month_data = ds_monthly.sel(date=time_step, method='nearest')

        # Convert the xarray data to a DataFrame for merging with the GeoDataFrame
        df_month = month_data.to_dataframe().reset_index()

        # Merge the flood percentage data with the GeoDataFrame
        gdf_with_data = gdf.merge(df_month[['region', 'flood_percent']], left_on='ADM2_NAME', right_on='region', how='left')

        # Create a plot for the current time step
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        gdf_with_data.plot(column='flood_percent', ax=ax, cmap='viridis', legend=True, vmin=flood_percent_min, vmax=flood_percent_max)

        # Set the title for the plot
        ax.set_title(f'Flooding Percentage on {time_step.values}', fontsize=16)
        ax.set_axis_off()

        # Save the current frame to a file
        frame_path = f'{output_dir}/frame_{time_step.values}.png'
        plt.savefig(frame_path)
        plt.close(fig)  # Close the figure to free memory

        # Append the file path to the frames list
        frames.append(frame_path)

    # Create GIF from the frames using imageio.v2
    with imageio.get_writer(output_gif, mode='I', duration=frame_duration) as writer:
        for frame in frames:
            image = imageio.imread(frame)
            writer.append_data(image)

    print(f"GIF created successfully as {output_gif} with each frame lasting {frame_duration} seconds!")

'''# Example usage of the function with frame_duration specified
create_flooding_gif(
    shapefile_dir='/Users/aws/Documents/personal_projects/african_parks/regional_shape_files/example.shp',
    data_dir='/Users/aws/Documents/personal_projects/african_parks/data/final_data_reduced/final_data.nc',
    frame_duration=3  # Each frame will last 3 seconds
)
'''

def analyze_flooding_trends_and_create_gif(shapefile_dir, data_dir, output_gif='flooding_trend_combined.gif', avg_r2_map_path='avg_r2_map.png', output = False):
    """
    Analyze flooding trends and create a GIF for both normalized slope (slope divided by mean flooding percentage)
    and R² scores for each region and month. Additionally, calculate the average R² score per region and save it as a map.

    Parameters:
    - shapefile_dir: Path to the shapefile containing region geometries.
    - data_dir: Path to the xarray dataset containing flooding data.
    - output_gif: Name of the output GIF file showing combined normalized slope and R² plots (default: 'flooding_trend_combined.gif').
    - avg_r2_map_path: Path to save the map showing average R² score per region.
    """

    # Create colour scale for normalized slope plots
    norm = TwoSlopeNorm(vmin=-0.001, vcenter=0, vmax=0.001)  # Adjusted for normalized slope values

    # Load geometries (GeoDataFrame)
    gdf = gpd.read_file(shapefile_dir)

    # Load the dataset (xarray Dataset)
    ds = xr.open_dataset(data_dir)

    # Convert the 'date' column to datetime type
    ds['date'] = pd.to_datetime(ds['date'].values, format='%Y_%m_%d')

    # Resample the data by month, taking the mean for each month
    ds_monthly = ds.resample(date="1ME").mean()
    ds_monthly['flood_percent'] = ds_monthly.flooded_pixels / ds_monthly.total_pixels

    # Convert the dataset to a Pandas DataFrame for easier manipulation
    df = ds_monthly.to_dataframe().reset_index()

    # Calculate the mean flooding percentage for each region across all months
    mean_flooding_by_region = df.groupby('region')['flood_percent'].mean().reset_index()
    mean_flooding_by_region.columns = ['region', 'mean_flood_percent']

    # Create output directory for the frames
    output_dir = 'frames'
    os.makedirs(output_dir, exist_ok=True)

    # Store slopes and R² scores for each region across all months to compute the average later
    r2_scores_by_region = {region: [] for region in df['region'].unique()}
    slopes_by_region = {region: [] for region in df['region'].unique()}

    # Store file paths of generated frames
    frames = []

    # Iterate over each unique month (January to December)
    for month in range(1, 13):
        # Store the normalized slopes and R² scores for each region in this specific month
        region_slopes = []
        region_r2 = []

        # Iterate over each region
        for region in df['region'].unique():
            region_data = df[df['region'] == region]

            # Filter data for the specific month (all Januaries, all Februaries, etc.)
            monthly_data = region_data[region_data['date'].dt.month == month]

            # Ensure there's enough data within the month to perform linear regression
            if len(monthly_data) > 1:
                # Use time (in days since start) as the x variable and flood_percent as the y variable
                X = (monthly_data['date'] - monthly_data['date'].min()).dt.days.values.reshape(-1, 1)
                y = monthly_data['flood_percent'].values

                valid = ~np.isnan(y)
                X = X[valid, :]
                y = y[valid]

                # Fit a linear regression model
                model = LinearRegression()
                model.fit(X, y)

                # Get regression statistics
                slope = model.coef_[0]
                r_squared = r2_score(y, model.predict(X))

                # Append R² to the region's list of R² scores
                r2_scores_by_region[region].append(r_squared)

                # Get the mean flooding percentage for this region across all months
                mean_flood_percent = mean_flooding_by_region.loc[mean_flooding_by_region['region'] == region, 'mean_flood_percent'].values[0]

                # Normalize the slope by dividing by the mean flooding percentage
                normalized_slope = slope / mean_flood_percent if mean_flood_percent != 0 else np.nan

                slopes_by_region[region].append(normalized_slope)

                # Store the region, normalized slope, and R² score information
                region_slopes.append({'region': region, 'slope': normalized_slope})
                region_r2.append({'region': region, 'r2': r_squared})

        # Convert the region slopes and R² to DataFrames for merging
        slopes_df = pd.DataFrame(region_slopes)
        r2_df = pd.DataFrame(region_r2)

        # Merge the normalized slope and R² data with the GeoDataFrame
        gdf_with_slopes = gdf.merge(slopes_df[['region', 'slope']], left_on='ADM2_NAME', right_on='region', how='left')
        gdf_with_r2 = gdf.merge(r2_df[['region', 'r2']], left_on='ADM2_NAME', right_on='region', how='left')

        # Create a plot for the current month (side-by-side)
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))  # Create two subplots side by side

        # Normalized slope plot on the left
        gdf_with_slopes.plot(column='slope', ax=axes[0], cmap='coolwarm', legend=True, norm=norm)
        axes[0].set_title(f'Normalized Flooding Trend (Slope / Mean) by Region - Month: {month}', fontsize=16)
        axes[0].set_axis_off()

        # R² plot on the right
        gdf_with_r2.plot(column='r2', ax=axes[1], cmap='viridis', legend=True, vmin=0, vmax=1)
        axes[1].set_title(f'R² Score by Region - Month: {month}', fontsize=16)
        axes[1].set_axis_off()

        # Save the current frame to a file
        frame_path = f'{output_dir}/frame_{month}.png'
        plt.savefig(frame_path)
        plt.close(fig)  # Close the figure to free memory

        # Append the file path to the frames list
        frames.append(frame_path)

    # Create GIF from the frames using imageio.v2
    with imageio.get_writer(output_gif, mode='I', duration=2) as writer:
        for frame in frames:
            image = imageio.imread(frame)
            writer.append_data(image)

    print(f"GIF created successfully as {output_gif}!")

    # Calculate the average R² score for each region
    avg_r2_by_region = {region: np.mean(r2_scores) for region, r2_scores in r2_scores_by_region.items()}

    # Convert average R² scores to DataFrame for merging
    avg_r2_df = pd.DataFrame(list(avg_r2_by_region.items()), columns=['region', 'avg_r2'])

    # Merge average R² scores with GeoDataFrame
    gdf_with_avg_r2 = gdf.merge(avg_r2_df, left_on='ADM2_NAME', right_on='region', how='left')

    # Create a map for the average R² score per region
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    gdf_with_avg_r2.plot(column='avg_r2', ax=ax, cmap='viridis', legend=True, vmin=0, vmax=1)

    # Customize and save the plot
    ax.set_title('Average R² Score by Region', fontsize=16)
    ax.set_axis_off()
    plt.savefig(avg_r2_map_path)
    plt.close()

    print(f"Average R² map saved as {avg_r2_map_path}!")

    if output == True:
        return slopes_by_region


'''# Example usage
data = analyze_flooding_trends_and_create_gif(
    shapefile_dir='/Users/aws/Documents/personal_projects/african_parks/regional_shape_files/example.shp',
    data_dir='/Users/aws/Documents/personal_projects/african_parks/data/final_data_reduced/final_data.nc',
    avg_r2_map_path='average_r2_map.png',
    output = True
)'''

ds = xr.open_dataset('/Users/aws/Downloads/VIIRS-Flood-5day-GLB001_v1r0_blend_s201201210032250_e201201242319340_c202205182351447.nc')

print(ds['WaterDetection'].TypeDescription)