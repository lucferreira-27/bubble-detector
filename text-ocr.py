import os
import zipfile
import requests
import base64
import json
import logging
from PIL import Image
from config import config
from datetime import datetime
from tqdm import tqdm

VISION_API_URL = 'https://vision.googleapis.com/v1/images:annotate'
VISION_API_KEY = config["VISION_API_KEY"]
OUTPUT_FOLDER = 'outputs/text-ocr/'

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def encode_image(image_path):
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('UTF-8')


def process_image(image_path):
    image_content = encode_image(image_path)
    
    payload = {
        'requests': [
            {
                'image': {'content': image_content},
                'features': [{'type': 'TEXT_DETECTION'}]
            }
        ]
    }

    response = requests.post(VISION_API_URL, params={'key': VISION_API_KEY}, json=payload)
    data = response.json()
    texts = data['responses'][0].get('textAnnotations', None)
    if(texts == None):
        return None
    results = {
        'image_name': os.path.basename(image_path),
        'image_dimensions': Image.open(image_path).size,
        'text_blocks': [{'vertices': [{'x': v.get('x', 0), 'y': v.get('y', 0)} for v in text['boundingPoly']['vertices']],
                        'text': text['description']} for text in texts]

    }

    return results


def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def save_results_to_file(file_path, results):
    with open(file_path, 'w') as f:
        json.dump(results, f, indent=4)


def process_folder(input_folder):
    if not os.path.isdir(input_folder):
        raise ValueError("Input folder not found")

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')

    for filename in tqdm(os.listdir(input_folder), desc="Processing images"):
        filepath = os.path.join(input_folder, filename)

        if filename.endswith(".cbz"):
            cbz_folder_name = os.path.join(OUTPUT_FOLDER, os.path.splitext(filename)[0])
            create_folder_if_not_exists(cbz_folder_name)

            with zipfile.ZipFile(filepath, 'r') as cbz_file:
                for name in cbz_file.namelist():
                    if name.lower().endswith(('.jpg', '.jpeg','.png')):
                        temp_img_path = os.path.join(cbz_folder_name, f'temp_{name}')
                        with open(temp_img_path, 'wb') as temp_img:
                            temp_img.write(cbz_file.read(name))
                        results = process_image(temp_img_path)
                        if(results == None):
                            logging.info(f"Skiping {name} because not text found")
                            os.remove(temp_img_path)
                            continue
                        os.remove(temp_img_path)
                        output_path = os.path.join(cbz_folder_name, f'{name}_{timestamp}_results.json')
                        save_results_to_file(output_path, results)
                        logging.info(f"Saved results for {name} to {output_path}")

        elif filename.lower().endswith(".jpg"):
            results = process_image(filepath)

            output_path = os.path.join(OUTPUT_FOLDER, f'{filename}_{timestamp}_results.json')
            save_results_to_file(output_path, results)
            logging.info(f"Saved results for {filename} to {output_path}")


if __name__ == "__main__":
    input_folder = "./images"
    process_folder(input_folder)
