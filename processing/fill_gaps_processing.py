import xarray as xr
import numpy as np
import os 
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline

def process_data():

    root = os.path.join(os.getcwd(), 'data')

    for year in ['2013']:

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
                    if '97' in file:
                        continue
                    
                    file = os.path.join(day_dir, file)
                    df = xr.open_dataset(file)

                # Define bounding box
                lon_min = 25.0
                lon_max = 30.0
                lat_min = 5.0
                lat_max = 10.0

                # Clip dataset
                clipped_df = df.sel(lon=slice(lon_min, lon_max), lat=slice(lat_max, lat_min))

                # Extract 'WaterDetection' values as a numpy array
                water_detection_values = clipped_df['WaterDetection'].values

                water_binary = ((water_detection_values >= 100) & (water_detection_values <= 200)).astype(float)

                # Construct the filename
                save_folder='/Users/aws/Documents/personal_projects/african_parks/terra_modis_data/viirs'
                filename = f"{year}_{month}_{day}.npy"

                # Save the numpy array to the specified folder
                np.save(f"{save_folder}/{filename}", water_binary)

                '''plt.imshow(water_binary, cmap='gray', interpolation='none')
                plt.colorbar(label='Water Detection (1=Yes, 0=No)')
                plt.title('Binary Water Detection Plot')
                plt.xlabel('Longitude Index')
                plt.ylabel('Latitude Index')
                plt.show()'''
                    

def reproject(image, final_x, final_y):
    # Check if the input image has multiple bands
    if image.ndim != 3:
        raise ValueError("Input image must have 3 dimensions (bands, width, height)")

    width, height, bands = image.shape
    
    # Define the original grid coordinates
    x = np.arange(width)
    y = np.arange(height)
    
    # Define the new grid coordinates for resizing
    x_new = np.linspace(0, width - 1, final_x)
    y_new = np.linspace(0, height - 1, final_y)
    
    reprojected_image = np.zeros((bands, final_x, final_y))

    # Iterate over each band and reproject
    for band in range(bands):
        single_band_image = image[:,:,band]
        
        # Create a RectBivariateSpline for the single band image
        spline = RectBivariateSpline(x, y, single_band_image)
        
        # Evaluate the spline on the new grid
        reprojected_image[band] = spline(x_new, y_new)

    print(reprojected_image.shape)

    return reprojected_image

def test_projection(path):
    # List all files in the directory
    file = os.listdir(path)[0]

    path = os.path.join(path, file)

    array = np.load(path)

    # Check if array is loaded
    if array is not None:
        
        # Define new shape for reprojection
        new_shape = (1400, 1800)
        
        # Reproject the array
        new_array = reproject(array, new_shape[0], new_shape[1])
        
        print(new_array.shape)

        bands = array.shape[2]
        
        for band in range(bands):
            # Get min and max values for consistent color scaling
            vmin = min(array[:, :, band].min(), new_array[band].min())
            vmax = max(array[:, :, band].max(), new_array[band].max())
            
            # Plotting
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))
            
            # Plot original array band
            im1 = axs[0].imshow(array[:, :, band], cmap='viridis', vmin=vmin, vmax=vmax)
            axs[0].set_title(f'Original Array - Band {band + 1}')
            
            # Plot reprojected array band
            im2 = axs[1].imshow(new_array[band], cmap='viridis', vmin=vmin, vmax=vmax)
            axs[1].set_title(f'Reprojected Array - Band {band + 1}')
            
            # Add a color bar to the figure for reference
            fig.colorbar(im1, ax=axs, orientation='horizontal', fraction=0.05, pad=0.1)

            plt.tight_layout()
            plt.show()

        # Print shapes for verification
        print(f"Original array shape: {array.shape}")
        print(f"Reprojected array shape: {new_array.shape}")

    else:
        print("No files found in the directory.")

def reproject_terra_modis(aqua_location, terra_location):
    aqua_save_folder = '/Users/aws/Documents/personal_projects/african_parks/terra_modis_data/aqua_reproj'
    terra_save_folder = '/Users/aws/Documents/personal_projects/african_parks/terra_modis_data/terra_reproj'

    # Create directories if they don't exist
    os.makedirs(aqua_save_folder, exist_ok=True)
    os.makedirs(terra_save_folder, exist_ok=True)

    for file in os.listdir(aqua_location):
        path = os.path.join(aqua_location, file)
        array = np.load(path)
        new_array = reproject(array, 1483, 1482)
        print(new_array.shape)
        np.save(os.path.join(aqua_save_folder, file), new_array)

    for file in os.listdir(terra_location):
        path = os.path.join(terra_location, file)
        array = np.load(path)
        new_array = reproject(array, 1483, 1482)
        np.save(os.path.join(terra_save_folder, file), new_array)


reproject_terra_modis('/Users/aws/Documents/personal_projects/african_parks/terra_modis_data/aqua', '/Users/aws/Documents/personal_projects/african_parks/terra_modis_data/terra')
    
#test_projection('/Users/aws/Documents/personal_projects/african_parks/terra_modis_data/aqua')
