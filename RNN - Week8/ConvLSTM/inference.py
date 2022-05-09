import os
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
import pandas as pd
import pickle

from utils.ucf_dataset import *
from utils.model import DecoderRNN, EncoderCNN
from utils.prediction import CRNN_final_prediction
from config import *


if __name__ == "__main__":

    action_names = ["Archery", "BaseballPitch", "Biking"]

    # convert labels -> category
    le = LabelEncoder()
    le.fit(action_names)

    # convert category -> 1-hot
    action_category = le.transform(action_names).reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(action_category)

    actions = []
    fnames = os.listdir(TEST_DATA_PATH)
    all_names = []
    for f in fnames:
        loc1 = f.find("v_")
        loc2 = f.find("_g")
        actions.append(f[(loc1 + 2) : loc2])

        all_names.append(f)

    # list all data files
    all_X_list = all_names  # all video file names
    all_y_list = labels2cat(le, actions)  # all video labels

    # data loading parameters
    use_cuda = torch.cuda.is_available()  # check if GPU exists
    device = torch.device("cuda" if use_cuda else "cpu")  # use CPU or GPU
    params = {"batch_size": BATCH_SIZE, "shuffle": True, "num_workers": 4, "pin_memory": True} if use_cuda else {}

    transform = transforms.Compose(
        [transforms.Resize([img_width, img_height]), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )

    selected_frames = np.arange(BEGIN_FRAME, END_FRAME, SKIP_FRAME).tolist()

    # reset data loader
    all_data_params = {"batch_size": BATCH_SIZE, "shuffle": False, "num_workers": 4, "pin_memory": True} if use_cuda else {}
    all_data_loader = data.DataLoader(Dataset_CRNN(TEST_DATA_PATH, all_X_list, all_y_list, selected_frames, transform=transform), **all_data_params)

    # reload CRNN model
    cnn_encoder = EncoderCNN(
        img_x=img_width, img_y=img_height, fc_hidden1=CNN_fc_hidden1, fc_hidden2=CNN_fc_hidden2, drop_p=dropout_p, CNN_embed_dim=CNN_embed_dim
    ).to(device)
    rnn_decoder = DecoderRNN(
        CNN_embed_dim=CNN_embed_dim, h_RNN_layers=RNN_hidden_layers, h_RNN=RNN_hidden_nodes, h_FC_dim=RNN_FC_dim, drop_p=dropout_p, num_classes=TOTAL_CLASSES
    ).to(device)

    cnn_encoder.load_state_dict(torch.load(os.path.join(SAVE_MODEL_PATH, "cnn_encoder_epoch60.pth")))
    rnn_decoder.load_state_dict(torch.load(os.path.join(SAVE_MODEL_PATH, "rnn_decoder_epoch60.pth")))
    print("CRNN model reloaded!")

    # make all video predictions by reloaded model
    print("Predicting all {} videos:".format(len(all_data_loader.dataset)))
    # all_y_pred = CRNN_final_prediction([cnn_encoder, rnn_decoder], device, all_data_loader)
    # print(all_y_pred)
    CRNN_final_prediction([cnn_encoder, rnn_decoder], device, all_data_loader)

    # # write in pandas dataframe
    # df = pd.DataFrame(data={"filename": fnames, "y": cat2labels(le, all_y_list), "y_pred": cat2labels(le, all_y_pred)})
    # df.to_pickle("./UCF101_videos_prediction.pkl")  # save pandas dataframe
    # # pd.read_pickle("./all_videos_prediction.pkl")
    # print("video prediction finished!")
