# https://discuss.pytorch.org/t/error-in-python-s-multiprocessing-library/31355/43
# https://discuss.pytorch.org/t/pytorch-equivalence-to-sparse-softmax-cross-entropy-with-logits-in-tensorflow/18727/2
# https://discuss.pytorch.org/t/multiclass-segmentation/54065/5
# https://discuss.pytorch.org/t/training-semantic-segmentation/49275/4

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from utils.dataset import CustomDataset
from models.model import UNet  # original version
from models.stable_unet.model import Stable_UNet  # stable version
from config import Config


def train(epochs, trainLoader, num_classes, criterion):
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, factor=0.1, verbose=True)

    mean_losses = []
    print("[INFO] Training is started.")
    for epoch in range(epochs):
        running_loss = []
        loop = tqdm(enumerate(trainLoader), total=len(trainLoader))

        for idx, (image, mask) in loop:
            image, mask = image.to(device), mask.to(device)
            outputs = model(image)

            loss = criterion(outputs, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss.append(loss.item())
            mean_loss = sum(running_loss) / len(running_loss)

            loop.set_description(f"Epoch: [{epoch + 1}/{epochs}]")
            loop.set_postfix(batch_loss=loss.item(), mean_loss=mean_loss, lr=optimizer.param_groups[0]["lr"])

        if len(mean_losses) >= 1:
            if mean_loss < min(mean_losses):
                print("[INFO] Model saved.")
                torch.save(model.state_dict(), "model.pth")

        mean_losses.append(mean_loss)
        scheduler.step(mean_loss)


if __name__ == "__main__":

    cfg = Config()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_dataset = CustomDataset(cfg.images_path, cfg.mask_paths, input_size=cfg.input_size)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=8)

    if cfg.model_type == "stable_version":
        model = Stable_UNet(in_channels=1, out_channels=64, n_class=cfg.num_classes, kernel_size=3, padding=1, stride=1)  # stable version
        criterion = nn.NLLLoss()
    elif cfg.model_type == "standard_version":
        model = UNet(num_classes=cfg.num_classes)  # unsable version
        criterion = nn.NLLLoss()
    model = model.to(device)

    # if cfg.model_type == "stable" > 2:
    #     criterion = nn.NLLLoss()
    # else:
    #     criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    train(epochs=cfg.num_epochs, trainLoader=train_loader, num_classes=cfg.num_classes, criterion=criterion)
    print("[INFO] Training is ended.")
