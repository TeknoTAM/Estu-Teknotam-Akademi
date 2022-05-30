
# Usage
- First store class images with seperate folder in data/train directory 
- There should be a separate file for each class
- Change parameters of config file
- Use torch2trt.py for tensorrt-onnx-tensorrt converter
- Use trt_inference.py to inference with tensorrt model engine

Complete train on custom dataset with torchvision pretrained models. 
Generate TensorRT engine optimized for the target platform with torch models.
Do inference with tensorrt engine.

---

# Tested with 
- torch 1.9.0
- torchvision 0.10.0
- tensorrt 8.2.3.0
- onnx 1.9.0
- onnxruntime 1.10.0
- pycuda 2021.1


- nvidia-driver 470.82.01
- cuda 11.4 update 3
- cudnn 8.2.4.15-1+cuda11.4(runtime and developer library)
- nvidia gtx 1050 ti

Follow tensorrt installation steps. 
- https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html
- Tensorrt Debian and RPM installations require that the CUDA toolkit and cuDNN have also been installed using Debian or RPM packages.

