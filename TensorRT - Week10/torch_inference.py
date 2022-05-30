import torch
from PIL import Image
import os
from natsort import natsorted
from models.models import initialize_model
import torchvision.transforms as transforms
import time

from classification_config import *


def predict(model, img):

    img = img.unsqueeze(0).cuda()
    out = model(img)

    prob = torch.softmax(out, dim=1)[0].cpu().detach().numpy()
    # prob = torch.nn.functional.softmax(out)

    pred = out.argmax().cpu().numpy()
    return pred, prob, out


if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transforms = transforms.Compose(
        [
            transforms.Resize(size=(INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    model = initialize_model(MODEL_NAME, NUM_CLASSES, PRETRAINED, INPUT_SIZE, FEATURE_EXTRACT)
    model.load_state_dict(torch.load(TORCH_MODEL_PATH))
    model.to(device)
    model.eval()

    test_path = TEST_DATA_DIR
    test_images = natsorted(os.listdir(test_path))

    for idx, image in enumerate(test_images):
        img = Image.open(test_path + image)

        img = data_transforms(img)
        img.to(device)
        with torch.no_grad():
            since = time.time()
            pred, prob, out = predict(model, img)
        # prob = prob.astype(np.float)
        # prob = np.round(prob,4)

        print(f"Image name: {test_path + image}")
        print(f"Raw outputs: {out}")
        print(f"Prob: {prob}, Pred: {pred}")
        print("Total inference time: ", time.time() - since)
        print("**************************\n")
