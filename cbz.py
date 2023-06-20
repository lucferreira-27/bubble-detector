import zipfile

def read_images(path="./images"):
    pass

def save_panels(output="./output/panels"):
    pass

def unzip_cbz(input, output):
    # Create a zipfile object from the input file
    zf = zipfile.ZipFile(input, 'r')
    # Extract all the files to the output directory
    zf.extractall(output)
    # Close the zipfile object
    zf.close()

    