import torch
import time
from loguru import logger

from classification_config import *
from utils.torch2onnx import torch2onnx_v2, onnx_pytorch_matching
from utils.onnx2engine import build_engine, test_engine


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    """TORCH 2 ONNX"""
    logger.info("Torch model is converting to onnx...")

    torch2onnx_v2(MODEL_NAME, NUM_CLASSES, PRETRAINED, INPUT_SIZE, FEATURE_EXTRACT, TORCH_MODEL_PATH, ONNX_MODEL_PATH, device)

    logger.info("Compare ONNX Runtime and Pytorch Results")

    onnx_pytorch_matching(MODEL_NAME, NUM_CLASSES, PRETRAINED, INPUT_SIZE, FEATURE_EXTRACT, TORCH_MODEL_PATH, ONNX_MODEL_PATH, device)

    # time.sleep(1)

    """ONNX TO ENGINE"""

    logger.info("Tensor engine model is building.")
    engine = build_engine(ONNX_MODEL_PATH, ENGINE_MODEL_PATH)  # convert onnx to engine

    logger.info("Engine model is testing...")
    test_engine(engine, INPUT_SIZE)  # test engine model over random inputs
