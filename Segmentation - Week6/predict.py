import os
from natsort import natsorted
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import torchvision
import cv2
import PIL


from models.model import UNet  # original version
from models.stable_unet.model import Stable_UNet  # stable version
from config import Config


def predict_mask(output, image):

    prediction = torch.argmax(output, 1)
    print("Prediction shape: ", prediction.shape)
    prediction = prediction.permute(1, 2, 0)  # CHW -> HWC,(1024,1024,1)
    print("Prediction shape after permute: ", prediction.shape)
    prediction = prediction.cpu().numpy()

    mask = prediction.astype("uint8")
    # reverse of label_color map in dataset script
    label_color_map = {0: 0, 1: 255}

    for k in label_color_map:
        mask[mask == k] = label_color_map[k]

    cv2.imshow("image", image.astype(np.uint8))
    cv2.imshow("mask", mask)
    cv2.waitKey(0)


def show_probability_map(output):

    # slice output channels of prediction, show probability map for each classes
    output = output.cpu()
    # prob = F.softmax(output,1)
    prob = torch.exp(output)  # we're using log_softmax in model, so apply torch.exp to get probabilities
    prob_imgs = torchvision.utils.make_grid(prob.permute(1, 0, 2, 3))
    plt.imshow(prob_imgs.permute(1, 2, 0))
    plt.show()

    # MORE THAN ONE BATCH, probability map for each classes
    # prob = F.softmax(output, 1)
    # prob = torch.exp(output)
    # for p in prob:
    #     prob_imgs = torchvision.utils.make_grid(p.unsqueeze(1))
    #     plt.imshow(prob_imgs.permute(1, 2, 0))
    #     plt.show()


if __name__ == "__main__":
    cfg = Config()
    to_tensor = transforms.ToTensor()
    to_gray = transforms.Grayscale()

    images_path = "./DATA/hiphop_data/images/"
    images = natsorted(os.listdir(images_path))

    for image in images:

        image = Image.open(images_path + image)
        image = image.resize((256, 256), resample=PIL.Image.NEAREST)
        image = to_gray(image)
        image = np.array(image)
        t_image = to_tensor(image)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if cfg.model_type == "stable_version":
            model = Stable_UNet(in_channels=1, out_channels=64, n_class=cfg.num_classes, kernel_size=3, padding=1, stride=1)  # stable version
        elif cfg.model_type == "standard_version":
            model = UNet(num_classes=cfg.num_classes)  # unsable version
        model = model.to(device)
        model.load_state_dict(torch.load("./model.pth"))
        model.eval()

        with torch.no_grad():
            t_image = t_image.unsqueeze(0)  # (1,1,1024,1024)
            t_image = t_image.to(device)
            print(t_image.shape)
            output = model(t_image)  # (1,3,1024,1024), 3 is number of classes, mapping must be number_classes - 1: 0,1,2

        predict_mask(output, image)
        # show_probability_map(output)

        # SIDE NOT
        # x = torch.randn(1, 3, 24, 24)
        # output = F.log_softmax(x, 1) #last process of model
        # print('LogProbs: ', output)
        # print('Probs: ', torch.exp(output))
        # print('Prediction: ', torch.argmax(output, 1))
