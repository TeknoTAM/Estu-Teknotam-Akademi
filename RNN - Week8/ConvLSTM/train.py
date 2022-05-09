import os
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.utils.data as data
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
import pickle

from utils.ucf_dataset import *
from utils.model import EncoderCNN, DecoderRNN
from config import *


def train(log_interval, model, device, train_loader, optimizer, epoch):
    # set model as training mode
    cnn_encoder, rnn_decoder = model
    cnn_encoder.train()
    rnn_decoder.train()

    losses = []
    scores = []
    N_count = 0  # counting total trained sample in one epoch
    for batch_idx, (X, y) in enumerate(train_loader):
        # distribute data to device
        X, y = X.to(device), y.to(device).view(
            -1,
        )

        N_count += X.size(0)

        optimizer.zero_grad()
        output = rnn_decoder(cnn_encoder(X))  # output has dim = (batch, number ocf classes)

        loss = F.cross_entropy(output, y)
        losses.append(loss.item())

        # to compute accuracy
        y_pred = torch.max(output, 1)[1]  # y_pred != output

        y_gt = y.cpu().data.squeeze().numpy()
        y_pred_cpu = y_pred.cpu().data.squeeze().numpy()
        step_score = accuracy_score([y_gt], [y_pred_cpu])
        scores.append(step_score)  # computed on CPU

        loss.backward()
        optimizer.step()

        # show information
        if (batch_idx + 1) % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch + 1, N_count, len(train_loader.dataset), 100.0 * (batch_idx + 1) / len(train_loader), loss.item()
                )
            )

    return losses, scores


def validation(model, device, optimizer, test_loader):
    # set model as testing mode
    cnn_encoder, rnn_decoder = model
    cnn_encoder.eval()
    rnn_decoder.eval()

    test_loss = 0
    all_y = []
    all_y_pred = []
    with torch.no_grad():
        for X, y in test_loader:
            # distribute data to device
            X, y = X.to(device), y.to(device).view(
                -1,
            )

            output = rnn_decoder(cnn_encoder(X))

            loss = F.cross_entropy(output, y, reduction="sum")
            test_loss += loss.item()  # sum up batch loss
            y_pred = output.max(1, keepdim=True)[1]  # (y_pred != output) get the index of the max log-probability

            # collect all y and y_pred in all batches
            all_y.extend(y)
            all_y_pred.extend(y_pred)

    test_loss /= len(test_loader.dataset)

    # compute accuracy
    all_y = torch.stack(all_y, dim=0)
    all_y_pred = torch.stack(all_y_pred, dim=0)
    test_score = accuracy_score(all_y.cpu().data.squeeze().numpy(), all_y_pred.cpu().data.squeeze().numpy())

    # show information
    print("\nTest set ({:d} samples): Average loss: {:.4f}, Accuracy: {:.2f}%\n".format(len(all_y), test_loss, 100 * test_score))

    # save Pytorch models of best record
    torch.save(cnn_encoder.state_dict(), os.path.join(SAVE_MODEL_PATH, "cnn_encoder_epoch{}.pth".format(epoch + 1)))  # save spatial_encoder
    torch.save(rnn_decoder.state_dict(), os.path.join(SAVE_MODEL_PATH, "rnn_decoder_epoch{}.pth".format(epoch + 1)))  # save motion_encoder
    torch.save(optimizer.state_dict(), os.path.join(SAVE_MODEL_PATH, "optimizer_epoch{}.pth".format(epoch + 1)))  # save optimizer
    print("Epoch {} model saved!".format(epoch + 1))

    return test_loss, test_score


if __name__ == "__main__":
    # Detect devices
    use_cuda = torch.cuda.is_available()  # check if GPU exists
    device = torch.device("cuda" if use_cuda else "cpu")  # use CPU or GPU

    # load UCF101 actions names
    # with open(ACTION_NAME_PATH, "rb") as f:
    #     action_names = pickle.load(f)

    action_names = ["Archery", "BaseballPitch", "Biking"]
    # convert labels -> category
    le = LabelEncoder()
    le.fit(action_names)

    # convert category -> 1-hot
    action_category = le.transform(action_names).reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(action_category)

    # example
    # y = ["Archery", "BaseballPitch", "Biking"]
    # y_onehot = labels2onehot(enc, le, y)
    # y2 = onehot2labels(le, y_onehot)
    # print("y_onehot: ", y_onehot)
    # print("y2: ", y2)

    actions = []
    fnames = os.listdir(DATA_PATH)

    all_names = []
    for f in fnames:
        loc1 = f.find("v_")
        loc2 = f.find("_g")
        actions.append(f[(loc1 + 2) : loc2])

        all_names.append(f)

    # list all data files
    all_X_list = all_names  # all video file names
    all_y_list = labels2cat(le, actions)  # all video labels

    # train, test split
    train_list, test_list, train_label, test_label = train_test_split(all_X_list, all_y_list, test_size=0.25, random_state=42)

    transform = transforms.Compose(
        [transforms.Resize([img_width, img_height]), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )

    selected_frames = np.arange(BEGIN_FRAME, END_FRAME, SKIP_FRAME).tolist()
    train_set = Dataset_CRNN(DATA_PATH, train_list, train_label, selected_frames, transform=transform)
    valid_set = Dataset_CRNN(DATA_PATH, test_list, test_label, selected_frames, transform=transform)

    params = {"batch_size": BATCH_SIZE, "shuffle": True, "num_workers": 4, "pin_memory": True} if use_cuda else {}
    train_loader = data.DataLoader(train_set, **params)
    valid_loader = data.DataLoader(valid_set, **params)

    # Create model
    cnn_encoder = EncoderCNN(
        img_x=img_width, img_y=img_height, fc_hidden1=CNN_fc_hidden1, fc_hidden2=CNN_fc_hidden2, drop_p=dropout_p, CNN_embed_dim=CNN_embed_dim
    ).to(device)

    rnn_decoder = DecoderRNN(
        CNN_embed_dim=CNN_embed_dim, h_RNN_layers=RNN_hidden_layers, h_RNN=RNN_hidden_nodes, h_FC_dim=RNN_FC_dim, drop_p=dropout_p, num_classes=TOTAL_CLASSES
    ).to(device)

    crnn_params = list(cnn_encoder.parameters()) + list(rnn_decoder.parameters())
    optimizer = torch.optim.Adam(crnn_params, lr=LR)

    # record training process
    epoch_train_losses = []
    epoch_train_scores = []
    epoch_test_losses = []
    epoch_test_scores = []

    # start training
    for epoch in range(TOTAL_EPOCH):
        # train, test model
        train_losses, train_scores = train(PRINT_INTERVAL, [cnn_encoder, rnn_decoder], device, train_loader, optimizer, epoch)
        epoch_test_loss, epoch_test_score = validation([cnn_encoder, rnn_decoder], device, optimizer, valid_loader)
