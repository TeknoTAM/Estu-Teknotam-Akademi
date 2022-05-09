import json
import os
import cv2
import numpy as np
import torch
from torchvision import transforms
import selectivesearch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_json(img_dir):
    json_file = os.path.join(img_dir, "via_region_data.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for idx, v in enumerate(imgs_anns.vg3alues()):
        record = {}
        
        filename = os.path.join(img_dir, v["filename"])
        height, width = cv2.imread(filename).shape[:2]
        
        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
      
        annos = v["regions"]
        objs = []
        for _, anno in annos.items():
            assert not anno["region_attributes"]
            anno = anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "category_id": 1,
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
def preprocess_image(img):
    img = torch.tensor(img).permute(2,0,1)
    img = normalize(img)
    return img.to(device).float()
def decode(_y):
    _, preds = _y.max(-1)
    return preds


def extract_candidates(img):
    img_lbl, regions = selectivesearch.selective_search(img, scale=200, min_size=100)
    img_area = np.prod(img.shape[:2])
    candidates = []
    for r in regions:
        if r['rect'] in candidates: continue
        if r['size'] < (0.05*img_area): continue
        if r['size'] > (1*img_area): continue
        x, y, w, h = r['rect']
        candidates.append(list(r['rect']))
    return candidates
def extract_iou(boxA, boxB, epsilon=1e-5):
    x1 = max(boxA[0], boxB[0])
    y1 = max(boxA[1], boxB[1])
    x2 = min(boxA[2], boxB[2])
    y2 = min(boxA[3], boxB[3])
    width = (x2 - x1)
    height = (y2 - y1)
    if (width<0) or (height <0):
        return 0.0
    area_overlap = width * height
    area_a = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    area_b = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    area_combined = area_a + area_b - area_overlap
    iou = area_overlap / (area_combined+epsilon)
    return iou

def get_deltas(src_boxes, target_boxes):
    src_boxes = np.asarray(src_boxes)
    target_boxes = np.asarray(target_boxes)
    src_widths = src_boxes[2] - src_boxes[0]
    src_heights = src_boxes[3] - src_boxes[1]
    src_ctr_x = src_boxes[0] + 0.5 * src_widths
    src_ctr_y = src_boxes[1] + 0.5 * src_heights

    target_widths = target_boxes[2] - target_boxes[0]
    target_heights = target_boxes[3] - target_boxes[1]
    target_ctr_x = target_boxes[0] + 0.5 * target_widths
    target_ctr_y = target_boxes[1] + 0.5 * target_heights


    dx = (target_ctr_x - src_ctr_x) / src_widths
    dy = (target_ctr_y - src_ctr_y) / src_heights
    dw = np.log(target_widths / src_widths)
    dh = np.log(target_heights / src_heights)

    deltas = np.asarray([dx,dy,dw,dh])
    return deltas