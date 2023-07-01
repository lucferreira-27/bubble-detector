from segment import run_model2annotations # Run model to find speech bubbles in manga page def run_model2annotations(img_dir, save_dir, save_json=False):
from panels import get_panels_labels # Code to find panels in manga page def extract_panels(image_dir):
from cbz import unzip_cbz
import os
import shutil
# Define the input and output directories
input_dir = './images/volumes/' # The directory where the cbz files are stored
output_dir = './manga_pages' # The directory where the extracted manga pages will be saved
annotation_dir = './output/annotations' # The directory where the speech bubble annotations will be saved
panel_dir = './output/panels' # The directory where the panel images will be saved

# Import the shutil module


# Loop through all the cbz files in the input directory
for file in os.listdir(input_dir):
    # Check if the file is a cbz file
    if file.endswith('.cbz') or file.endswith('.zip') or file.endswith('.rar'):
        # Get the file name without the extension
        file_name = os.path.splitext(file)[0]
        # Create a subdirectory for the extracted manga pages with the same name as the cbz file
        output_subdir = os.path.join(output_dir, file_name)
        # Unzip the cbz file to the output subdirectory
        unzip_cbz(os.path.join(input_dir, file), output_subdir)
        # Run the model to find speech bubbles in the manga pages and save the annotations as json files
        # run_model2annotations(output_subdir,f'{annotation_dir}/{os.path.basename(file)}' , save_json=True)
        # Extract the panels from the manga pages and save them as images
        get_panels_labels(output_subdir, f'{panel_dir}/{os.path.basename(file)}')
        # Delete the output subdirectory
        # shutil.rmtree(output_dir)

