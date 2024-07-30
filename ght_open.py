import rasterio
import matplotlib.pyplot as plt

# Load the .hgt file
hgt_file = './NASADEM/s03w055.num'

with rasterio.open(hgt_file) as src:
    elevation_data = src.read(1)

# Plot the elevation data
plt.imshow(elevation_data, cmap='terrain')
plt.colorbar(label='Elevation (meters)')
plt.title('DEM from .hgt file')
plt.show()
