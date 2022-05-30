import torch
import torch.nn as nn
from torchvision import models
from torchsummary import summary

# If we'll feature extract not finetuning, feature extract should be True.
def set_parameter_requires_grad(model,feature_extract):
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name,num_classes,pretrained,input_size,featureExtract = False):

    device = torch.device("cuda")

    if "resnet" in model_name:
        if model_name == "resnet18": model = models.resnet18(pretrained=pretrained)
        elif model_name == "resnet34": model = models.resnet34(pretrained=pretrained)
        elif model_name == "resnet50": model = models.resnet50(pretrained=pretrained)
        elif model_name == "resnet101": model = models.resnet101(pretrained=pretrained)
        elif model_name == "resnet152": model = models.resnet152(pretrained=pretrained)
        elif model_name == "wide-resnet50": model = models.wide_resnet50_2(pretrained=pretrained)
        elif model_name == "wide-resnet101": model = models.wide_resnet101_2(pretrained=pretrained)

        set_parameter_requires_grad(model,feature_extract=featureExtract)
        num_filters = model.fc.in_features
        model.fc = nn.Linear(in_features = num_filters, out_features = num_classes)
        #print(model)
        #summary(model.to(device),(3,input_size[0],input_size[1]))

    elif "vgg" in model_name:
        if model_name == "vgg19": model = models.vgg19(pretrained=pretrained)
        elif model_name == "vgg16": model = models.vgg16(pretrained=pretrained)
        elif model_name == "vgg13": model = models.vgg13(pretrained=pretrained)
        elif model_name == "vgg11": model = models.vgg11(pretrained=pretrained)
        elif model_name == "vgg19_bn": model = models.vgg19_bn(pretrained=pretrained)
        elif model_name == "vgg16_bn": model = models.vgg16_bn(pretrained=pretrained)
        elif model_name == "vgg13_bn": model = models.vgg13_bn(pretrained=pretrained)
        elif model_name == "vgg11_bn": model = models.vgg11_bn(pretrained=pretrained)
        
        set_parameter_requires_grad(model,feature_extract=featureExtract)
        num_filters = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features = num_filters, out_features = num_classes)
        print(model)
        summary(model.to(device),(3,input_size[0],input_size[1]))

    elif model_name == "alexnet":

        model = models.alexnet(pretrained=pretrained)
        set_parameter_requires_grad(model,feature_extract=featureExtract)
        num_filters = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features = num_filters, out_features = num_classes)
        print(model)
        summary(model.to(device),(3,input_size[0],input_size[1]))

   
    elif model_name == "squeezenet":
        """SqueezeNet 1.1"""
        model = models.squeezenet1_1(pretrained=pretrained)
        set_parameter_requires_grad(model,feature_extract=featureExtract)
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model.num_classes = num_classes
        print(model)
        summary(model.to(device),(3,input_size[0],input_size[1]))

    elif model_name == "densenet":
        """ Densenet"""
        
        model = models.densenet121(pretrained=pretrained)
        set_parameter_requires_grad(model,feature_extract=featureExtract)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
        print(model)
        summary(model.to(device),(3,input_size[0],input_size[1]))

    elif model_name == "mobilenet_v2":
        """MobileNet V2"""
        
        model = models.mobilenet_v2(pretrained = pretrained)
        set_parameter_requires_grad(model,feature_extract=featureExtract)
        num_filters = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features = num_filters, out_features = num_classes)
        print(model)
        summary(model.to(device),(3,input_size[0],input_size[1]))

    elif model_name == "mobilenet_v3_small":
        """MobileNet V3 Small"""
        
        model = models.mobilenet_v3_small(pretrained = pretrained)
        set_parameter_requires_grad(model,feature_extract=featureExtract)
        num_filters = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features = num_filters, out_features = num_classes)
        print(model)
        summary(model.to(device),(3,input_size[0],input_size[1]))

    elif model_name == "mobilenet_v3_large":
        """MobileNet V3 Large"""
        
        model = models.mobilenet_v3_large(pretrained = pretrained)
        set_parameter_requires_grad(model,feature_extract=featureExtract)
        num_filters = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features = num_filters, out_features = num_classes)
        print(model)
        summary(model.to(device),(3,input_size[0],input_size[1]))

    elif model_name == "googlenet":
        """GoogleNet"""
        
        model = models.googlenet(pretrained=pretrained)
        set_parameter_requires_grad(model,feature_extract=featureExtract)
        num_filters = model.fc.in_features
        model.fc = nn.Linear(in_features = num_filters, out_features = num_classes)
        print(model)
        summary(model.to(device),(3,input_size[0],input_size[1]))

    elif model_name == "shufflenet_v2_x1.0":
        """ShuffleNet V2 x1.0"""
        model = models.shufflenet_v2_x1_0(pretrained=pretrained)
        set_parameter_requires_grad(model,feature_extract=featureExtract)
        num_filters = model.fc.in_features
        model.fc = nn.Linear(in_features = num_filters, out_features = num_classes)
        print(model)
        summary(model.to(device),(3,input_size[0],input_size[1]))
    
    elif model_name == "shufflenet_v2_x1.5":
        """ShuffleNet V2 x1.5"""
        # pretrained = True is not supported

        model = models.shufflenet_v2_x1_5(pretrained=False)
        set_parameter_requires_grad(model,feature_extract=featureExtract)
        num_filters = model.fc.in_features
        model.fc = nn.Linear(in_features = num_filters, out_features = num_classes)
        print(model)
        summary(model.to(device),(3,input_size[0],input_size[1]))

    elif model_name == "shufflenet_v2_x2.0":
        """ShuffleNet V2 x1.5"""
        # pretrained = True is not supported
        model = models.shufflenet_v2_x2_0(pretrained=False)
        set_parameter_requires_grad(model,feature_extract=featureExtract)
        num_filters = model.fc.in_features
        model.fc = nn.Linear(in_features = num_filters, out_features = num_classes)
        print(model)
        summary(model.to(device),(3,input_size[0],input_size[1]))
    
    elif model_name == "inception":
        """Inception V3"""
        # inception just accepts 299x299 shape
        model = models.inception_v3(pretrained=pretrained)
        set_parameter_requires_grad(model,feature_extract=featureExtract)
        num_filters = model.fc.in_features
        model.fc = nn.Linear(in_features = num_filters, out_features = num_classes)
        print(model)
        if input_size[0] < 299 or input_size[1] < 299:
            print("Inception takes minimum 299x299 tensors.")
            exit()
        summary(model.to(device),(3,299,299))

    else:
        print("Invalid model name, exiting...")
        exit()

    return model