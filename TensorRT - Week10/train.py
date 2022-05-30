import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from tqdm import tqdm
from loguru import logger

from models.models import initialize_model
from utils.calc_mean_std import calc_mean_std
from classification_config import *


def train_model(model_name, model, train_data, input_size, num_epochs, batch_size, lr, num_classes):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    model.to(device)

    mean, std = calc_mean_std(train_data, input_size, batch_size)

    data_transforms = transforms.Compose(
        [
            transforms.Resize(size=(input_size)),
            # transforms.ColorJitter(brightness=0.6,contrast=0.8,saturation=0.7,hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        ]
    )

    trainset = torchvision.datasets.ImageFolder(train_data, transform=data_transforms)
    train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)

    # optimizer = torch.optim.SGD(params_to_update, lr = lr, momentum = momentum)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.05, verbose=True)

    if num_classes == 1:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    logger.info("Training started.")

    mean_losses = []
    for epoch in range(num_epochs):
        running_loss = []
        # lossMeter.reset()
        loop = tqdm(enumerate(train_loader), total=len(train_loader))
        for batch_idx, (data, target) in loop:
            data, target = data.to(device), target.to(device)

            if model_name == "inception":
                output, aux = model(data)
            else:
                output = model(data)

            optimizer.zero_grad()

            if num_classes == 1:
                loss = criterion(output, target.float())
            else:
                loss = criterion(output, target)

            loss.backward()
            optimizer.step()

            running_loss.append(loss.item())
            mean_loss = sum(running_loss) / len(running_loss)

            loop.set_description(f"Epoch: [{epoch + 1}/{num_epochs}]")
            loop.set_postfix(batch_loss=loss.item(), mean_loss=mean_loss, lr=optimizer.param_groups[0]["lr"])

        if len(mean_losses) >= 1:
            if mean_loss < min(mean_losses):
                logger.info("Model saved.")
                torch.save(model.state_dict(), TORCH_MODEL_PATH)

        mean_losses.append(mean_loss)
        scheduler.step(mean_loss)


if __name__ == "__main__":

    model = initialize_model(MODEL_NAME, NUM_CLASSES, PRETRAINED, INPUT_SIZE, FEATURE_EXTRACT)

    train_model(MODEL_NAME, model, TRAIN_DATA_DIR, INPUT_SIZE, NUM_EPOCHS, BATCH_SIZE, LR, NUM_CLASSES)

    logger.info("Training ended.")
