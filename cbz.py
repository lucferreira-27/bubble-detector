import zipfile
import os
import shutil
def read_images(path="./images"):
    pass

def save_panels(output="./output/panels"):
    pass

def unzip_cbz(input, output):
    # Create a zipfile object from the input file
    print(input)
    zf = zipfile.ZipFile(input, 'r')
    # Loop through the files in the zipfile
    for name in zf.namelist():
        # Check if the file is an image
        if name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            # Extract the image to the output directory
            zf.extract(name, output)
            # Get the image path 
            image_path = os.path.join(output, name)
            # Check if the image is in a subdirectory
            if '/' in name:
                # Get the image filename
                image_name = os.path.basename(name)
                # Move the image to the output directory
                shutil.move(image_path, os.path.join(output, image_name))
                # Remove the subdirectory
                shutil.rmtree(os.path.dirname(image_path))
    # Close the zipfile object
    zf.close()



    