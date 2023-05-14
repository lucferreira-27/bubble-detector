import config
import os
import zipfile
from inference import model2annotations
from tqdm import tqdm
import requests

def download_models(model_path='data/comictextdetector.pt'):
    if os.path.exists(model_path):
        print('ComicTextDetector model already exists.')
        return

    model_dir = os.path.dirname(model_path)
    if not os.path.exists(model_dir):
        print(f"Creating directory {model_dir}")
        os.makedirs(model_dir)
    
    # Download ComicTextDetector model
    model_url = 'https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.2.1/comictextdetector.pt'
    print('Downloading ComicTextDetector model...')
    response = requests.get(model_url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True,ncols=50)
    with open(model_path, 'wb') as f:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            f.write(data)
    progress_bar.close()
    print('Download completed.')
    
def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def run_model2annotations(img_dir, model_path, save_dir, save_json=False):
    # Convert images to annotations using ComicTextDetector model
    print('Running ComicTextDetector model...')
    
    temp_images = []

# Create temporary images
    for filename in os.listdir(img_dir):
        if filename.endswith(".cbz"):
            cbz_folder_name = os.path.join(save_dir, os.path.splitext(filename)[0])
            create_folder_if_not_exists(cbz_folder_name)

            with zipfile.ZipFile(os.path.join(img_dir, filename), 'r') as cbz_file:
                for name in cbz_file.namelist():
                    if name.lower().endswith(('.jpg', '.jpeg','.png')):
                        temp_img_path = os.path.join(cbz_folder_name, f'temp_{name}')
                        with open(temp_img_path, 'wb') as temp_img:
                            temp_img.write(cbz_file.read(name))
                        temp_images.append(temp_img_path)
        else:
            model2annotations(model_path, img_dir, save_dir, save_json=save_json)

    # Process temporary images with model2annotations
    for temp_img_path in temp_images:
        cbz_folder_name = os.path.dirname(temp_img_path)
        model2annotations(model_path, cbz_folder_name, cbz_folder_name, save_json=save_json)
        os.remove(temp_img_path)

    print('\nAnnotations saved to', save_dir)


if __name__ == '__main__':
    
    model_path = 'model/comictextdetector.pt'
    
    download_models(model_path)

    # Define input and output directories
    img_dir = 'images'
    save_dir = 'outputs/speech-bubbles/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Run model2annotations
    run_model2annotations(img_dir, model_path, save_dir, save_json=True)