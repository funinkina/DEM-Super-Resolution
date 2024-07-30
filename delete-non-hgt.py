import os


def delete_non_hgt_files(folder):
    # Iterate over all the files in the specified folder
    for item in os.listdir(folder):
        item_path = os.path.join(folder, item)

        # Check if the item is a file and does not end with .hgt extension
        if os.path.isfile(item_path) and not item.endswith(".hgt"):
            os.remove(item_path)
            print(f"Deleted: {item}")


# Specify the folder where the files are located
folder = "./hgt_files"

# Call the function to delete non .hgt files
delete_non_hgt_files(folder)
