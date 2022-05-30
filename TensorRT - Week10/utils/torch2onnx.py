import onnx
import onnxruntime
import torch
import numpy as np

from models.models import initialize_model

def torch2onnx_v2(model_name,num_classes,pretrained,input_size,feature_extract,torch_model_path,onnx_model_path,device):
    torch_model = initialize_model(model_name,num_classes,pretrained, input_size,feature_extract)

    dummy_input = torch.randn(1,3,input_size[0],input_size[1],requires_grad = True).to(device)

    torch_model.load_state_dict(torch.load(torch_model_path))   
    torch_model = torch_model.to(device).eval()
    output = torch_model(dummy_input)

    torch.onnx.export(
            torch_model,
            (dummy_input,),
            onnx_model_path,
            example_outputs = output,
            export_params = True,
            opset_version = 11,
            verbose = True,
            do_constant_folding = True,
            input_names = ["input"],
            output_names = ["output"],
            #dynamic_axes = {'input' : {1: 'sequence'},'output': {1:'sequence'}}
        )

def torch2onnx_v1(model_name,num_classes,pretrained,input_size,feature_extract,torch_model_path,onnx_model_path,device):
    torch_model = initialize_model(model_name,num_classes,pretrained, input_size,feature_extract)

    dummy_input = torch.randn(1,3,input_size[0],input_size[1],requires_grad = True).to(device)
    torch_model.load_state_dict(torch.load(torch_model_path))
    torch_model = torch_model.to(device).eval()

    torch.onnx.export(torch_model,dummy_input,onnx_model_path,opset_version = 11)

"""Pytorch and Onnx Matching"""
def onnx_pytorch_matching(model_name,num_classes,pretrained,input_size,feature_extract,torch_model_path,onnx_model_path,device):
      
    torch_model = initialize_model(model_name,num_classes,pretrained, input_size,feature_extract)
    torch_model.load_state_dict(torch.load(torch_model_path))
    torch_model.eval()
    
    dummy_input = torch.ones(1,3,input_size[0],input_size[1])
    torch_out = torch_model(dummy_input)
    ort_session = onnxruntime.InferenceSession(onnx_model_path)
    
    #compute onnxruntime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.detach().numpy()}
    ort_outs = ort_session.run(None,ort_inputs)
    
    print("Onnx outs: ",ort_outs[0])
    print("Pytorch outs: ",torch_out)

    #compare ONNX Runtime and Pytorch Results
    #np.testing.assert_allclose(torch_out[0].detach().numpy(),ort_outs[0],rtol=1e-03, atol=1e-05)
    #print("Exported model has been tested with ONNXRuntime, and the result looks good!")



