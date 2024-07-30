import rasterio
from PIL import Image
import numpy as np

def hgt_to_jpg(hgt_path, jpg_path):
    # Load the .hgt file
    with rasterio.open(hgt_path) as src:
        elevation_data = src.read(1)

    # Normalize the data to the range 0-255 for JPEG
    min_val = np.min(elevation_data)
    max_val = np.max(elevation_data)
    normalized_data = ((elevation_data - min_val) / (max_val - min_val) * 255).astype(np.uint8)

    # Convert to Image object
    img = Image.fromarray(normalized_data, mode='L')

    # Save as .jpg
    img.save(jpg_path)

# Example usage
hgt_path = './NASADEM/s03w055.hgt'
jpg_path = 'output_file.jpg'
hgt_to_jpg(hgt_path, jpg_path)
