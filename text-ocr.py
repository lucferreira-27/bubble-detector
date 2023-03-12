import os
import requests
import base64
import json
from datetime import datetime
from PIL import Image
from config import config

# Define the endpoint for the Vision API's text detection feature
default_path = "./images/One Piece v1-008.jpg"
url = 'https://vision.googleapis.com/v1/images:annotate'

# Set up the request payload
payload = {
    'requests': [
        {
            'image': {
                'content': base64.b64encode(open(default_path, 'rb').read()).decode('UTF-8')
            },
            'features': [
                {
                    'type': 'TEXT_DETECTION'
                }
            ]
        }
    ]
}

# Set up the API key for authentication
params = {
    'key': config["VISION_API_KEY"]
}

# Send the request to the Vision API
response = requests.post(url, params=params, json=payload)

# Parse the response to extract the OCR text
data = json.loads(response.text)
texts = data['responses'][0]['textAnnotations']

# Save the OCR text and vertices to a JSON file
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
filename = f"{default_path.split('/')[-1].split('.')[0]}_{timestamp}.json"

results = {'image_name': os.path.basename(default_path), 'image_dimensions': Image.open(default_path).size, 'text_blocks': []}
for text in texts:
    result = {'vertices': [], 'text': text['description']}
    vertices = text['boundingPoly']['vertices']
    for v in vertices:
        result['vertices'].append({'x': v['x'], 'y': v['y']})
    results['text_blocks'].append(result)

if not os.path.exists('outputs/text-ocr'):
    os.makedirs('outputs/text-ocr',exist_ok=True)
    
with open('outputs/text-ocr/' + filename , 'w') as f:
    json.dump(results, f, indent=4)
    
print(f'OCR results saved to outputs/text-ocr/{filename}')
