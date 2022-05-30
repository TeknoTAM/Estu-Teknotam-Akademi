import torchvision.transforms as transforms

MODEL_NAME = "resnet50"
INPUT_SIZE = (200, 200)  # (height,width)
BATCH_SIZE = 4
NUM_EPOCHS = 100
NUM_CLASSES = 2
LR = 1e-4
MOMENTUM = 0.9

PRETRAINED = True  # pretrained weights that train on Imagenet dataset
FEATURE_EXTRACT = False  # When True, Freeze pretrained layers. Just use for feature extract.When False, update all params of model.


TRAIN_DATA_DIR = "./data/train"
TEST_DATA_DIR = "./data/test/"

CHECKPOINT_DIR = "./checkpoints/"
TORCH_MODEL_PATH = CHECKPOINT_DIR + "model_" + MODEL_NAME + ".pth"

ONNX_MODEL_PATH = CHECKPOINT_DIR + "onnx_models/" + MODEL_NAME + ".onnx"
ENGINE_MODEL_PATH = CHECKPOINT_DIR + "engine_models/" + MODEL_NAME + ".engine"
