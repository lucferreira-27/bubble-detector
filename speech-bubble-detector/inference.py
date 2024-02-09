from skimage import io
import json
from basemodel import TextDetBase, TextDetBaseDNN
import os.path as osp
from tqdm import tqdm
import numpy as np
import cv2
import torch
from pathlib import Path
import torch
from utils.yolov5_utils import non_max_suppression
from utils.db_utils import SegDetectorRepresenter
from utils.io_utils import imread, imwrite, find_all_imgs, NumpyEncoder
from utils.imgproc_utils import letterbox, xyxy2yolo, get_yololabel_strings
from utils.textblock import TextBlock, group_output, visualize_textblocks
from utils.textmask import refine_mask, refine_undetected_mask, REFINEMASK_INPAINT, REFINEMASK_ANNOTATION
from pathlib import Path
from typing import Union
import os
from image_utils import combine_images
from panel_segment import extract_panels,sort_by_read,join_panels_bubbles
from gemini import transcript
import random


def calculate_iou(box1, box2):
    
  # Calculate intersection area
  x1 = max(box1["xyxy"][0], box2["xyxy"][0])
  y1 = max(box1["xyxy"][1], box2["xyxy"][1])
  x2 = min(box1["xyxy"][2], box2["xyxy"][2])
  y2 = min(box1["xyxy"][3], box2["xyxy"][3])
  intersection = max(0, x2 - x1) * max(0, y2 - y1)

  # Calculate union area
  box1_area = (box1["xyxy"][2] - box1["xyxy"][0]) * (box1["xyxy"][3] - box1["xyxy"][1])
  box2_area = (box2["xyxy"][2] - box2["xyxy"][0]) * (box2["xyxy"][3] - box2["xyxy"][1])
  union = box1_area + box2_area - intersection

  # Calculate IoU
  iou = intersection / union
  return iou

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


def inicialize_model(model_path,device):
    if isinstance(img_dir_list, str):
        img_dir_list = [img_dir_list]
    cuda = torch.cuda.is_available()
    device = 'cuda' if cuda else 'cpu'
    model = TextDetector(model_path=model_path, input_size=1024, device=device, act='leaky')
    return model 


def model2annotations(model_path, img_dir_list, save_dir, save_json=False):
    pages = []
    if isinstance(img_dir_list, str):
        img_dir_list = [img_dir_list]
    cuda = torch.cuda.is_available()
    device = 'cuda' if cuda else 'cpu'
    model = TextDetector(model_path=model_path, input_size=1024, device=device, act='leaky')  
    imglist = []
    for img_dir in img_dir_list:
        imglist += find_all_imgs(img_dir, abs_path=True)
    for img_path in tqdm(imglist):
        imgname = os.path.basename(img_path)
        img = cv2.imread(img_path)
        im_h, im_w = img.shape[:2]
        imgExt = Path(imgname).suffix
        imname = imgname.replace(imgExt, '')
        
        image_save_dir = f'{save_dir}/{imname}/text'                          
        os.makedirs(image_save_dir, exist_ok=True) 
        _, _, merged_list = model(img, refine_mode=REFINEMASK_ANNOTATION, keep_undetected_mask=False)

        polys = []
        blk_int_xyxy = []
        blk_dict_list = []
        for blk in merged_list:
            polys += blk.lines
            blk_int_xyxy.append(blk.xyxy)
            blk_dict_list.append(blk.to_dict())

        blk_int_xyxy = merge_overlapping_boxes(blk_dict_list)
        
        # Draw bounding boxes on the image
        for i, bbox in enumerate(blk_int_xyxy):
            bbox_img = cut_text_blocks(img,bbox["xyxy"])
            print(f"{bbox['xyxy']} {i + 1}_{imgExt} {bbox['font_size']}")
            bbox_img_path = os.path.join(image_save_dir, f"{i + 1}_{imgExt}")
            bbox["filename"] = os.path.basename(bbox_img_path)
            cv2.imwrite(bbox_img_path, bbox_img)

        panels = extract_panels(img_path)
    
        folder_path = os.path.dirname(bbox_img_path)
        total_responses = start(folder_path)
        bubbles = []
        for bbox in blk_int_xyxy:
            if bbox["filename"] in total_responses:
                bubble = {  
                    "text":total_responses[bbox["filename"]]["text"],
                    "xyxy": bbox["xyxy"]
                }
                bubbles.append(bubble)      
                
        blocks = join_panels_bubbles(bubbles,panels)
        blocks = sort_by_read(blocks)
        #draw_bounding_boxes(io.imread(img_path),blocks)
        page = {
                        "blocks": blocks,
                        "image_path": img_path
        }
        pages.append(page)
    return pages
        
    

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

def preprocess_img(img, input_size=(1024, 1024), device='cpu', bgr2rgb=True, half=False, to_tensor=True):
    if bgr2rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_in, ratio, (dw, dh) = letterbox(img, new_shape=input_size, auto=False, stride=64)
    if to_tensor:
        img_in = img_in.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img_in = np.array([np.ascontiguousarray(img_in)]).astype(np.float32) / 255
        if to_tensor:
            img_in = torch.from_numpy(img_in).to(device)
            if half:
                img_in = img_in.half()
    return img_in, ratio, int(dw), int(dh)

def postprocess_mask(img: Union[torch.Tensor, np.ndarray], thresh=None):
    # img = img.permute(1, 2, 0)
    if isinstance(img, torch.Tensor):
        img = img.squeeze_()
        if img.device != 'cpu':
            img = img.detach_().cpu()
        img = img.numpy()
    else:
        img = img.squeeze()
    if thresh is not None:
        img = img > thresh
    img = img * 255
    # if isinstance(img, torch.Tensor):

    return img.astype(np.uint8)

def postprocess_yolo(det, conf_thresh, nms_thresh, resize_ratio, sort_func=None):
    det = non_max_suppression(det, conf_thresh, nms_thresh)[0]
    # bbox = det[..., 0:4]
    if det.device != 'cpu':
        det = det.detach_().cpu().numpy()
    det[..., [0, 2]] = det[..., [0, 2]] * resize_ratio[0]
    det[..., [1, 3]] = det[..., [1, 3]] * resize_ratio[1]
    if sort_func is not None:
        det = sort_func(det)

    blines = det[..., 0:4].astype(np.int32)
    confs = np.round(det[..., 4], 3)
    cls = det[..., 5].astype(np.int32)
    return blines, cls, confs

class TextDetector:
    lang_list = ['eng', 'ja', 'unknown']
    langcls2idx = {'eng': 0, 'ja': 1, 'unknown': 2}

    def __init__(self, model_path, input_size=1024, device='cpu', half=False, nms_thresh=0.35, conf_thresh=0.4, mask_thresh=0.3, act='leaky'):
        super(TextDetector, self).__init__()
        cuda = device == 'cuda'

        if Path(model_path).suffix == '.onnx':
            self.model = cv2.dnn.readNetFromONNX(model_path)
            self.net = TextDetBaseDNN(input_size, model_path)
            self.backend = 'opencv'
        else:
            self.net = TextDetBase(model_path, device=device, act=act)
            self.backend = 'torch'
        
        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        self.input_size = input_size
        self.device = device
        self.half = half
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.seg_rep = SegDetectorRepresenter(thresh=0.3)

    @torch.no_grad()
    def __call__(self, img, refine_mode=REFINEMASK_INPAINT, keep_undetected_mask=False):
        img_in, ratio, dw, dh = preprocess_img(img, input_size=self.input_size, device=self.device, half=self.half, to_tensor=self.backend=='torch')
        im_h, im_w = img.shape[:2]

        blks, mask, lines_map = self.net(img_in)

        resize_ratio = (im_w / (self.input_size[0] - dw), im_h / (self.input_size[1] - dh))
        blks = postprocess_yolo(blks, self.conf_thresh, self.nms_thresh, resize_ratio)

        if self.backend == 'opencv':
            if mask.shape[1] == 2:     # some version of opencv spit out reversed result
                tmp = mask
                mask = lines_map
                lines_map = tmp
        mask = postprocess_mask(mask)

        lines, scores = self.seg_rep(self.input_size, lines_map)
        box_thresh = 0.6
        idx = np.where(scores[0] > box_thresh)
        lines, scores = lines[0][idx], scores[0][idx]
        
        # map output to input img
        mask = mask[: mask.shape[0]-dh, : mask.shape[1]-dw]
        mask = cv2.resize(mask, (im_w, im_h), interpolation=cv2.INTER_LINEAR)
        if lines.size == 0 :
            lines = []
        else :
            lines = lines.astype(np.float64)
            lines[..., 0] *= resize_ratio[0]
            lines[..., 1] *= resize_ratio[1]
            lines = lines.astype(np.int32)
        blk_list = group_output(blks, lines, im_w, im_h, mask)
        mask_refined = refine_mask(img, mask, blk_list, refine_mode=refine_mode)
        mask_undetected,seg_blk_list = refine_undetected_mask(img, mask, mask_refined, blk_list, refine_mode=refine_mode)
        merged_list = blk_list + seg_blk_list
        return merged_list


if __name__ == '__main__':
    device = 'gpu'
    model_path = 'data/comictextdetector.pt'
    model_path = 'data/comictextdetector.pt.onnx'
    img_dir = r'data/examples'
    save_dir = r'data/output'
    model2annotations(model_path, img_dir, save_dir, save_json=True)
    #traverse_by_dict(img_dir, save_dir)