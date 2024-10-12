import boto3
import os
from tqdm import tqdm

# Initialize a session using Amazon S3
s3 = boto3.client('s3')

# Define the bucket name and prefix (folder)
bucket_name = 'noaa-jpss'
prefix = 'JPSS_Blended_Products/VFM_5day_GLB/NetCDF/'

# Function to download files with 'GLB096' or 'GLB097' in their names
def download_files_with_pattern(bucket, prefix, patterns=['GLB096', 'GLB097'], local_dir='./data/'):
    # List all objects in the bucket with the specified prefix
    paginator = s3.get_paginator('list_objects_v2')
    result_iterator = paginator.paginate(Bucket=bucket, Prefix=prefix)
    
    for page in tqdm(result_iterator, desc='Downloading', unit=' pages'):
        if 'Contents' in page:
            for obj in page['Contents']:
                key = obj['Key']
                if any(pattern in key for pattern in patterns) and key.endswith('.nc'):  # Check for patterns and NetCDF extension
                    # Extract relative path (remove bucket name and initial prefix)
                    relative_path = os.path.relpath(key, prefix)
                    # Construct local file path
                    local_file_path = os.path.join(local_dir, relative_path)
                    
                    # Ensure the directory structure exists locally
                    os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                    
                    # Download the file
                    s3.download_file(bucket, key, local_file_path)

# Call the function to download files
download_files_with_pattern(bucket_name, prefix)
