"""
This code is re-implementation of tensorrt7 for tensorrt 8.2.3
Includes of inference with tensorrt engine for classification tasks.
"""

import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import tensorrt as trt
import cv2
from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import time
from natsort import natsorted

from utils.onnx2engine import alloc_buf
from classification_config import *


def preprocess(frame):
    data_transforms = transforms.Compose(
        [
            transforms.Resize(size=(INPUT_SIZE)),
            # transforms.ColorJitter(brightness=0.6,contrast=0.8,saturation=0.7,hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        ]
    )

    PIL_image = Image.fromarray(frame.astype("uint8"), "RGB")
    input_data = data_transforms(PIL_image)

    batch_data = torch.unsqueeze(input_data, 0)  # torch.Size([1, 3, 100, 100])
    # frame = cv2.resize(frame,(INPUT_SIZE[1],INPUT_SIZE[0]))
    # frame = np.expand_dims(frame,0) # (1,h,w,3)
    # frame = np.moveaxis(frame,[1,2],[2,3]).astype(np.float32) # (1,3,h,w)
    return batch_data


def inference(engine, context, inputs, out_cpu, in_gpu, out_gpu, stream):
    # async version
    """with engine.create_execution_context() as context:  # cost time to initialize
    cuda.memcpy_htod_async(in_gpu, inputs, stream)
    context.execute_async(1, [int(in_gpu), int(out_gpu)], stream.handle, None)
    cuda.memcpy_dtoh_async(out_cpu, out_gpu, stream)
    stream.synchronize()"""

    # sync version
    cuda.memcpy_htod(in_gpu, inputs)
    context.execute(1, [int(in_gpu), int(out_gpu)])
    cuda.memcpy_dtoh(out_cpu, out_gpu)
    return out_cpu


if __name__ == "__main__":

    runtime = trt.Runtime(trt.Logger(trt.Logger.INFO))

    runtime = trt.Runtime(trt.Logger())
    with open(ENGINE_MODEL_PATH, "rb") as f:
        buffer = f.read()
        engine = runtime.deserialize_cuda_engine(buffer)

    context = engine.create_execution_context()

    """Inference on One Image"""
    # image = cv2.imread("./data/nobel_data/test/iyi/img-102.jpg")
    # input_frame = preprocess(image.copy())

    # in_cpu, out_cpu, in_gpu, out_gpu, stream = alloc_buf(engine)

    # host_input = np.array(input_frame.numpy(), dtype=np.float32, order='C')
    # host_output = inference(engine, context, host_input.reshape(-1), out_cpu, in_gpu, out_gpu, stream)

    # tensor = torch.Tensor(host_output) #torch.Size([2]) #<built-in method type of Tensor object at 0x7fa19189a700>
    # confidences = torch.nn.functional.softmax(tensor, dim=0).cpu().detach().numpy()

    # prediction = np.argmax(host_output)
    # print(f"Raw Output: {host_output}")
    # print(f"Confidences: {confidences}")
    # print(f"Pred: {prediction}")

    """Inference on Multiple Images """
    images = natsorted(os.listdir(TEST_DATA_DIR))

    for image in images:
        frame = cv2.imread(TEST_DATA_DIR + image)
        input_frame = preprocess(frame.copy())

        in_cpu, out_cpu, in_gpu, out_gpu, stream = alloc_buf(engine)

        host_input = np.array(input_frame.numpy(), dtype=np.float32, order="C")
        since = time.time()
        host_output = inference(engine, context, host_input.reshape(-1), out_cpu, in_gpu, out_gpu, stream)

        tensor = torch.Tensor(host_output)
        confidences = torch.nn.functional.softmax(tensor, dim=0).cpu().detach().numpy()

        prediction = np.argmax(host_output)

        print("Image name: ", TEST_DATA_DIR + image)
        print("Raw outputs: ", host_output)
        print(f"Confidences: {confidences}, Pred: {prediction}")
        print("Total inference time: ", time.time() - since)
        print("**************************\n")
