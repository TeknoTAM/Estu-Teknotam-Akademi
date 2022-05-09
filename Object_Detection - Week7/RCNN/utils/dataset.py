import torch
import numpy as np
import cv2
from tqdm import tqdm
from utils.utils import preprocess_image, extract_candidates, extract_iou, get_deltas

device = "cuda" if torch.cuda.is_available() else "cpu"


class RCNNDataset(torch.utils.data.Dataset):
    def __init__(self, fpaths, gtbbs, labels, deltas, rois):
        self.fpaths = fpaths
        self.gtbbs = gtbbs
        self.rois = rois
        self.labels = labels
        self.deltas = deltas

    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, ix):
        fpath = str(self.fpaths[ix])
        image = cv2.imread(fpath, 1)[..., ::-1]  # conver BGR to RGB
        H, W, _ = image.shape
        sh = np.array([W, H, W, H])
        gtbbs = self.gtbbs[ix]
        rois = self.rois[ix]
        bbs = (np.array(rois) * sh).astype(np.uint16)
        labels = self.labels[ix]
        deltas = self.deltas[ix]
        crops = [image[y:Y, x:X] for (x, y, X, Y) in bbs]
        return image, crops, bbs, labels, deltas, gtbbs, fpath

    def collate_fn(self, batch):
        input, rois, rixs, labels, deltas = [], [], [], [], []
        for ix in range(len(batch)):
            image, crops, image_bbs, image_labels, image_deltas, image_gt_bbs, image_fpath = batch[ix]
            crops = [cv2.resize(crop, (224, 224)) for crop in crops]
            crops = [preprocess_image(crop / 255.0)[None] for crop in crops]
            input.extend(crops)
            labels.extend(image_labels)
            deltas.extend(image_deltas)
        input = torch.cat(input).to(device)
        labels = torch.Tensor(labels).long().to(device)
        deltas = torch.Tensor(deltas).float().to(device)
        return input, labels, deltas


def get_data(ds):
    FPATHS, GTBBS, CLSS, DELTAS, ROIS, IOUS = [], [], [], [], [], []
    for ix, obj in enumerate(tqdm(ds)):
        if ix == 20:
            break
        fpath = obj["file_name"]
        labels = [i["category_id"] for i in obj["annotations"]]
        bbs = [i["bbox"] for i in obj["annotations"]]
        H, W = obj["height"], obj["width"]
        im = cv2.imread(fpath)
        candidates = extract_candidates(im)
        candidates = np.array([(x, y, x + w, y + h) for x, y, w, h in candidates])
        ious, rois, clss, deltas = [], [], [], []
        ious = np.array([[extract_iou(candidate, _bb_) for candidate in candidates] for _bb_ in bbs]).T
        for jx, candidate in enumerate(candidates):
            cx, cy, cX, cY = candidate
            candidate_ious = ious[jx]
            best_iou_at = np.argmax(candidate_ious)
            best_iou = candidate_ious[best_iou_at]
            best_bb = _x, _y, _X, _Y = bbs[best_iou_at]
            if best_iou >= 0.5:
                clss.append(labels[best_iou_at])
            elif best_iou <= 0.3:
                clss.append(0)
            else:
                continue
            delta = get_deltas([cx, cy, cX, cY], [_x, _y, _X, _Y])
            deltas.append(delta)
            rois.append(candidate / np.array([W, H, W, H]))
        FPATHS.append(fpath)
        IOUS.append(ious)
        ROIS.append(rois)
        CLSS.append(clss)
        DELTAS.append(deltas)
        GTBBS.append(bbs)
    return FPATHS, GTBBS, CLSS, DELTAS, ROIS
