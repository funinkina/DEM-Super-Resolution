import os
import zipfile


def extract_all_zip_files(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate over all the files in the input folder
    for item in os.listdir(input_folder):
        item_path = os.path.join(input_folder, item)

        # Check if the item is a zip file
        if zipfile.is_zipfile(item_path):
            with zipfile.ZipFile(item_path, "r") as zip_ref:
                # Extract all the contents into the output folder
                zip_ref.extractall(output_folder)
                print(f"Extracted: {item}")


# Specify the input folder containing the zip files
input_folder = "./NASADEM_HGT_001-20240730_105940"

# Specify the output folder where the contents will be extracted
output_folder = "./hgt_files"

# Call the function to extract all zip files
extract_all_zip_files(input_folder, output_folder)
