from osgeo import gdal


# Example usage
file_path = './hgt_files/n49e028.hgt'

dataset = gdal.Open(file_path)

geotransform = dataset.GetGeoTransform()


x_res = geotransform[1]
y_res = abs(geotransform[5])
print(f"Resolution: {x_res} degrees in x direction, {y_res} degrees in y direction")
