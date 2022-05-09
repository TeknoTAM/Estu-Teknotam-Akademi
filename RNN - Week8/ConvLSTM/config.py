DATA_PATH = "./Data/3_classes/"  # define UCF-101 RGB data path
TEST_DATA_PATH = "./Data/test_data/"
ACTION_NAME_PATH = "./UCF101actions.pkl"
SAVE_MODEL_PATH = "./checkpoints/"

# training parameters
TOTAL_CLASSES = 3  # number of target category
TOTAL_EPOCH = 60  # training epochs
BATCH_SIZE = 1
LR = 1e-4
PRINT_INTERVAL = 10  # interval for displaying training info

# Select which frame to begin & end in videos
BEGIN_FRAME, END_FRAME, SKIP_FRAME = 1, 29, 1

# EncoderCNN architecture parameters
CNN_fc_hidden1, CNN_fc_hidden2 = 1024, 768
CNN_embed_dim = 512  # latent dim extracted by 2D CNN
img_width, img_height = 256, 342  # resize video 2d frame size
dropout_p = 0.0  # dropout probability

# DecoderRNN architecture parameters
RNN_hidden_layers = 3
RNN_hidden_nodes = 512
RNN_FC_dim = 256
