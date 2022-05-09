import os
import cv2
import numpy as np
import torch
import random
import torchvision.transforms as transforms
from loguru import logger
from PIL import Image
from natsort import natsorted
from torchvision import transforms
from train import get_model_instance_segmentation


def predict(model, image, device):
    """
    model: fastrcnnpredictor and maskrcnnpredictor
    image: must be tensor. Also must be list because of model's network.
    device: cuda or cpu

    output(Dict): [{'scores': [], 'labels': [], boxes: [], masks: []}]
    """
    to_tensor = transforms.ToTensor()
    model.eval()
    image = to_tensor(image)
    image = image.to(device)
    image = [image]

    with torch.no_grad():
        outputs = model(image)
        output = outputs[0]

    labels = output["labels"].detach().cpu().numpy()
    scores = output["scores"].detach().cpu().numpy()
    boxes = output["boxes"].detach().cpu().numpy()
    masks = output["masks"].detach().cpu().numpy()
    logger.info(f"Scores: {scores}")
    logger.info(f"Labels: {labels}")
    return labels, scores, boxes, masks


def threshold_detection(labels, scores, boxes, masks, threshold=0.7):
    """
    Apply threshold for eliminate low confidence detections

    """

    # specify index by theresholding the scores
    threshold_indices = [idx for idx, score in enumerate(scores) if score > threshold]
    threshold_count = len(threshold_indices)

    # get thresholding boxes and labels
    final_scores = scores[:threshold_count]
    final_labels = labels[:threshold_count]
    final_boxes = boxes[:threshold_count]
    final_masks = masks[:threshold_count]
    return final_scores, final_labels, final_boxes, final_masks


def visualization(image, scores, labels, boxes, masks, mask_threshold=0.5):

    """

    image (np.array,uint8): original image
    scores: chosen scores(greater than threshold)
    labels = chosen labels
    boxes = chosen masks
    mask_threshold = threshold for pixel-wise segmentation classifier

    """

    alpha = 1
    beta = 0.6
    gamma = 0

    # loop over detections
    for i in range(len(masks)):
        mask = masks[i].transpose(1, 2, 0)  # N,H,W --> H,W,C
        mask = np.squeeze(mask, axis=2)  # delete channel for visualization

        red_map = np.zeros(shape=(mask.shape[0], mask.shape[1]), dtype=np.uint8)
        green_map = np.zeros(shape=(mask.shape[0], mask.shape[1]), dtype=np.uint8)
        blue_map = np.zeros(shape=(mask.shape[0], mask.shape[1]), dtype=np.uint8)

        color = COLORS[random.randrange(0, len(COLORS))]  # shape: (3,)
        red_map[mask > mask_threshold], green_map[mask > mask_threshold], blue_map[mask > mask_threshold] = color  # assign random color value for every channel
        segmentation_map = np.stack([red_map, green_map, blue_map], axis=2)  # stack channels

        image = cv2.addWeighted(image, alpha, segmentation_map, beta, gamma, image)  # add image and segmentation map for transparent visualization
        cv2.rectangle(image, (int(boxes[i][0]), int(boxes[i][1])), (int(boxes[i][2]), int(boxes[i][3])), color=color, thickness=2)
        # cv2.putText(image,labels[i],(boxes[i][0],boxes[i][1] - 10), cv2.FONT_HERSHEY_SIMPLEX,1,color,thickness=2)

    return image


if __name__ == "__main__":
    global COLORS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 2

    COLORS = np.random.uniform(0, 255, size=(255, 3))

    model = get_model_instance_segmentation(num_classes)
    model.load_state_dict(torch.load("./checkpoints/model.pth"))
    model.to(device)

    image_dir = "./PennFudanPed/PNGImages/"
    images = natsorted(os.listdir(image_dir))

    for image_name in images:
        orig_img = cv2.imread(image_dir + image_name)
        img = Image.fromarray(orig_img).convert("RGB")

        labels, scores, boxes, masks = predict(model, img, device)
        f_scores, f_labels, f_boxes, f_masks = threshold_detection(labels, scores, boxes, masks, threshold=0.3)
        image = visualization(orig_img, f_scores, f_labels, f_boxes, f_masks)

        cv2.imshow("Image", image)
        key = cv2.waitKey(0)

        if key == ord("d"):
            pass
        elif key == ord("q"):
            exit()
