import os 
import numpy as np
import cv2
import torch
import torchvision
from PIL import Image
from natsort import natsorted


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self,data_path):
        super().__init__()

        self.data_path = data_path
        self.images = natsorted(os.listdir(self.data_path))
        
        self.transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    def __len__(self):
        return len(self.images)

    def __getitem__(self,index):
        image_path = os.path.join(self.data_path + self.images[index])
        org_image = Image.open(image_path)

        image = org_image.resize((256,256))
        image = self.transform(image)

        if self.images[index].split("_")[0] == 'dog':
            target = 0
        else:
            target = 1

        return image,target



"""Debug"""    
dataset = CustomDataset(data_path="./data")
for image,target in dataset:
    print("Image shape: ",image.shape)
    print("Target shape: ",target.shape)
