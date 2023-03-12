import config
import os
import sys
import time
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


def run_model2annotations(img_dir, model_path, save_dir, save_json=False):
    # Convert images to annotations using ComicTextDetector model
    print('Running ComicTextDetector model...')
    frames = ['\\', '|', '/', '-']
    for i in range(20):
        sys.stdout.write('\r' + frames[i % len(frames)])
        sys.stdout.flush()
        time.sleep(0.1)
    model2annotations(model_path, img_dir, save_dir, save_json=save_json)
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