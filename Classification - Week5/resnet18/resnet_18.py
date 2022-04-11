import torch
from torch import nn
import torch.nn.functional as F


class Block(nn.Module):

    def __init__(self,in_channels,out_channels,downsample = False):
        super(Block,self).__init__()
        self.downsample = downsample

        if downsample: 
            self.basic_conv = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=2,padding=1)
            self.downsample_conv = nn.Conv2d(in_channels=in_channels,out_channels=out_channels, kernel_size=1,stride=2,padding=0)
        else:
            self.basic_conv = nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=3,stride=1,padding=1)

        self.basic_bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.bootleneck_conv = nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=3,stride=1,padding=1)
        self.bootleneck_bn = nn.BatchNorm2d(out_channels)

    def forward(self,x):
        residual = x

        out = self.basic_conv(x)
        out = self.basic_bn(out)
        out = self.relu(out)

        out = self.bootleneck_conv(out)
        out = F.dropout(out,0.5)
        out = self.bootleneck_bn(out)
        out = self.relu(out)
        
        if self.downsample:
            residual = self.downsample_conv(residual)
        
        out = out.clone() + residual
        return out
        


class Resnet18(nn.Module):
    def __init__(self,num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7,stride=2,padding=3)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.layer1 = Block(64,64)
        self.layer2 = Block(64,64)
        self.layer3 = Block(64,128,downsample=True)
        self.layer4 = Block(128,128)
        self.layer5 = Block(128,256,downsample=True)
        self.layer6 = Block(256,256)
        self.layer7 = Block(256,512,downsample=True)
        self.layer8 = Block(512,512)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        #self.fc = nn.Linear(512,1) # binary classification
    
    def forward(self,x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool1(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)

        out = self.avgpool(out)
        out = self.fc(out.view(out.shape[0],-1))

        return out
        




"""Debug"""
# model = Resnet18(num_classes=2)
# x =  torch.rand(size=(1,3,224,224))

# out = model(x)
# print("Output shape: ",out.shape)