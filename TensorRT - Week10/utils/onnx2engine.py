import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import tensorrt as trt
import numpy as np
import pycuda.driver as cuda 
import pycuda.autoinit
import time



def build_engine(onnx_model_path,engine_model_path):
    
    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
    explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(explicit_batch) as network, trt.OnnxParser(network, TRT_LOGGER) as parser: 
        #builder.max_workspace_size = 1 << 20 # deprecated with tensorrt 8

        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 20
        builder.max_batch_size = 1
        
        if builder.platform_has_fast_fp16:
            builder.fp16_mode = True

        with open(onnx_model_path, "rb") as f:
            parser.parse(f.read())  
        
        engine = builder.build_engine(network,config)
        with open(engine_model_path, "wb") as f:
            f.write(engine.serialize())
        return engine

def alloc_buf(engine):
    # host cpu mem
    h_in_size = trt.volume(engine.get_binding_shape(0))
    h_out_size = trt.volume(engine.get_binding_shape(1))
    h_in_dtype = trt.nptype(engine.get_binding_dtype(0))
    h_out_dtype = trt.nptype(engine.get_binding_dtype(1))
    in_cpu = cuda.pagelocked_empty(h_in_size, h_in_dtype)
    out_cpu = cuda.pagelocked_empty(h_out_size, h_out_dtype)
    # allocate gpu mem
    in_gpu = cuda.mem_alloc(in_cpu.nbytes)
    out_gpu = cuda.mem_alloc(out_cpu.nbytes)
    stream = cuda.Stream()
    return in_cpu, out_cpu, in_gpu, out_gpu, stream


def test_engine(engine, input_size):
    inputs = np.random.random((1,input_size[1],input_size[0],3)).astype(np.float32)
    context = engine.create_execution_context()

    # async version
    """with engine.create_execution_context() as context:  # cost time to initialize
        cuda.memcpy_htod_async(in_gpu, inputs, stream)
        context.execute_async(1, [int(in_gpu), int(out_gpu)], stream.handle, None)
        cuda.memcpy_dtoh_async(out_cpu, out_gpu, stream)
        stream.synchronize()"""

    # sync version
    for _ in range(5):
        in_cpu, out_cpu, in_gpu, out_gpu, stream = alloc_buf(engine)
        cuda.memcpy_htod(in_gpu, inputs)
        context.execute(1, [int(in_gpu), int(out_gpu)])
        cuda.memcpy_dtoh(out_cpu, out_gpu)
   
        t1 = time.time()
        print("SHAPE: ", out_cpu.shape)
        print("PREDICTION: ", out_cpu)
        print("cost time: ", time.time() - t1)

