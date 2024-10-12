# Load required packages
library(ncdf4)
library(raster)

# Open NetCDF file
ncfile <- nc_open("/Users/aws/Documents/personal_projects/african_parks/data/2015/01/01/VIIRS-Flood-5day-GLB096_v1r0_blend_s201412311116520_e201501011244370_c202206141506211.nc")


# Print file information
print(ncfile)

# List dimensions
#print(ncdim(ncfile))

# List variables
#print(ncvar(ncfile))

# Close NetCDF file
nc_close(ncfile) 
