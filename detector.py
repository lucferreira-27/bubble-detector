import config
import os
import zipfile
from inference import TextDetector,REFINEMASK_ANNOTATION
from utils.io_utils import find_all_imgs
from tqdm import tqdm
import requests
import json
import torch
import cv2
from pathlib import Path
from panel_segment import extract_panels,sort_by_read,join_panels_bubbles
from gemini import transcript
import shutil


def calculate_distance(box1, box2):

    # Calculate the distance between boxes (if they do not overlap)
    dx = max(0, max(box1["xyxy"][0], box2["xyxy"][0]) - min(box1["xyxy"][2], box2["xyxy"][2]))
    dy = max(0, max(box1["xyxy"][1], box2["xyxy"][1]) - min(box1["xyxy"][3], box2["xyxy"][3]))
    return max(dx, dy)

def merge_boxes(box1, box2):
    x1 = min(box1["xyxy"][0], box2["xyxy"][0])
    y1 = min(box1["xyxy"][1], box2["xyxy"][1])
    x2 = max(box1["xyxy"][2], box2["xyxy"][2])
    y2 = max(box1["xyxy"][3], box2["xyxy"][3])
    return {"xyxy":[x1, y1, x2, y2],"font_size": max(box1["font_size"], box2["font_size"])}

def remove_small_boxes(boxes, area_threshold):
    filtered_boxes = []
    for box in boxes:
        width = box["xyxy"][2] - box["xyxy"][0]
        height = box["xyxy"][3] - box["xyxy"][1]
        box["area"] = (width * height)
        if width * height >= area_threshold:
            filtered_boxes.append(box)
    return filtered_boxes
    

def merge_overlapping_boxes(boxes, iou_threshold=10):
    merged_boxes = remove_small_boxes(boxes,2500)
    i = 0
    while i < len(merged_boxes):
        j = 0
        while j < len(merged_boxes):
            box1 = merged_boxes[i]
            box2 = merged_boxes[j]
            
            if box1 == box2:
                j += 1
                continue
            dist = calculate_distance(box1, box2)

            if dist <= iou_threshold:
                merge_box = merge_boxes(box1,box2)
                #print(f"{box1['xyxy']}, {box2['xyxy']} = {merge_box['xyxy']}")
                merged_boxes.remove(box1)
                merged_boxes.remove(box2)
                merged_boxes.append(merge_box)
                j = 0  # Reset the inner loop
                i = 0
            else:
                j += 1  # Only increment j if no merge happened
        i += 1
    return merged_boxes


def cut_text_blocks(img, bbox, padding=0.1):
    x1, y1, x2, y2 = bbox
    h, w = img.shape[:2]
    pad_w = int((x2 - x1) * padding)
    pad_h = int((y2 - y1) * padding)
    x1 = max(0, x1 - pad_w)
    y1 = max(0, y1 - pad_h)
    x2 = min(w, x2 + pad_w)
    y2 = min(h, y2 + pad_h)
    return img[y1:y2, x1:x2]



def download_models(model_path='data/comictextdetector.pt.onnx'):
    if os.path.exists(model_path):
        print('[SYSTEM] -> ComicTextDetector model already exists.')
        return

    model_dir = os.path.dirname(model_path)
    if not os.path.exists(model_dir):
        print(f"[SYSTEM] -> Creating directory {model_dir}")
        os.makedirs(model_dir)
    
    # Download ComicTextDetector model
    model_url = 'https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.2.1/comictextdetector.pt.onnx'
    print('[SYSTEM] -> Downloading ComicTextDetector model...')
    response = requests.get(model_url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True,ncols=50)
    with open(model_path, 'wb') as f:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            f.write(data)
    progress_bar.close()
    print('[SYSTEM] -> Download completed.')
    
def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def initialize_model(model_path,device='cuda'):
    print("[TEXT DETECTOR] -> Initializing model...")
    cuda = torch.cuda.is_available()
    device = 'cuda' if cuda else 'cpu'
    model = TextDetector(model_path=model_path, input_size=1024, device=device, act='leaky')
    return model 


def find_images(images_dirs):
    print("[IMAGE FINDER] -> Finding images...")
    imglist = []
    if isinstance(images_dirs, str):
        images_dirs = [images_dirs]
    for img_dir in images_dirs:
        imglist += find_all_imgs(img_dir, abs_path=True)
    return imglist

def process_image(model, img_path, save_dir):
    print(f"[IMAGE PROCESSOR] -> Processing image: {img_path}")
    imgname = os.path.basename(img_path)
    img = cv2.imread(img_path)
    imgExt = Path(imgname).suffix 
    imname = imgname.replace(imgExt, '')
    image_save_dir = f'{save_dir}/{imname}'                          
    os.makedirs(image_save_dir, exist_ok=True) 
    merged_list = model(img, refine_mode=REFINEMASK_ANNOTATION)
    blk_dict_list = [blk.to_dict() for blk in merged_list]
    blk_int_xyxy = merge_overlapping_boxes(blk_dict_list)
    for i, bbox in enumerate(blk_int_xyxy):
        bbox_img = cut_text_blocks(img, bbox["xyxy"])
        bbox_img_path = os.path.join(image_save_dir, f"{i + 1}_{imgExt}")
        bbox["filename"] = os.path.basename(bbox_img_path)
        cv2.imwrite(bbox_img_path, bbox_img)
    return blk_int_xyxy, image_save_dir,img

def save_annotations(pages, save_dir,source_folder_name):
    print("[ANNOTATION SAVER] -> Saving annotations...")
    pages = json.loads(json.dumps(pages, default=int))
    with open(os.path.join(save_dir, f'{source_folder_name}.json'), 'w') as json_file:
        json.dump(pages, json_file,indent=4)
    print('[ANNOTATION SAVER] -> Annotations saved to', save_dir)


def delete_transcript_folder(folder_path):
    shutil.rmtree(folder_path)

def load_existing_annotations(save_dir, source_folder_name):
    annotations_path = os.path.join(save_dir, f'{source_folder_name}.json')
    if os.path.exists(annotations_path):
        with open(annotations_path, 'r') as json_file:
            return json.load(json_file)
    return []

def image_already_processed(pages, img_path):
    normalized_img_path = img_path.replace('\\', '/')
    for page in pages:
        if page["image_path"].replace('\\', '/') == normalized_img_path:
            return True
    return False

def check_all_images_processed(images_dir, save_dir):
    """
    Check if all images in the given directory have already been processed and annotations exist.
    
    :param images_dir: Directory containing images to check.
    :param save_dir: Directory where annotations are saved.
    :return: True if all images have been processed, False otherwise.
    """
    imglist = find_images(images_dir)
    source_folder_name = os.path.basename(images_dir)
    pages = load_existing_annotations(save_dir, source_folder_name)
    
    processed_images = {page["image_path"].replace('\\', '/') for page in pages}
    for img_path in imglist:
        normalized_img_path = img_path.replace('\\', '/')
        if normalized_img_path not in processed_images:
            return False
    return True

def run_model2annotations(images_dir, save_dir, save_json=False):
    print("[CONTROLER DETECTOR] -> Running ComicTextDetector model...")
    model_path = 'model/comictextdetector.pt.onnx'
    
    # Check if all images have already been processed before proceeding
    if check_all_images_processed(images_dir, save_dir):
        print("[INFO] -> All images have already been processed.")
        return  # Exit the function if all images are processed
    
    if not os.path.exists(model_path):
        download_models(model_path)
    model = initialize_model(model_path)
    
    imglist = find_images(images_dir)
    source_folder_name = os.path.basename(images_dir)
    pages = load_existing_annotations(save_dir, source_folder_name)  # Load existing annotations

    for img_path in tqdm(imglist):
        if image_already_processed(pages, img_path):  # Check if image is already processed
            print(f"[SKIPPING] -> {img_path} has already been processed.")
            continue  # Skip this image

        blk_int_xyxy, folder_path, img = process_image(model, img_path, save_dir)
        panels = extract_panels(img_path)
        total_responses = transcript(folder_path)
        delete_transcript_folder(folder_path)
        bubbles = []
        for bbox in blk_int_xyxy:
            if bbox["filename"] in total_responses:
                response = total_responses[bbox["filename"]]
                bubble = {
                    "gemini_block": response.get("gemini_block", False),
                    "text": response.get("text", ""),
                    "xyxy": bbox["xyxy"]
                }
                bubbles.append(bubble)
        blocks = join_panels_bubbles(bubbles, panels)
        blocks = sort_by_read(blocks)
        im_h, im_w = img.shape[:2]
        page = {
            "blocks": blocks,
            "image_path": img_path,
            "width": im_w,
            "height": im_h
        }
        pages.append(page)
        save_annotations(pages, save_dir, source_folder_name)  # Save after each image is processed
