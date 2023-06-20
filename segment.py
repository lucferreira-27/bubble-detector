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

def run_model2annotations(img_dir, save_dir, save_json=False):
    # Convert images to annotations using ComicTextDetector model
    os.makedirs(save_dir, exist_ok=True)
    model_path = 'model/comictextdetector.pt'
    if not os.path.exists(model_path):
        download_models(model_path)
    print('Running ComicTextDetector model...')
    model2annotations(model_path, img_dir, save_dir, save_json=save_json)
    print('\nAnnotations saved to', save_dir)