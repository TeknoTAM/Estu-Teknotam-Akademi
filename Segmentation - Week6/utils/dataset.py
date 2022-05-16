import os
from torchvision import transforms
from torch.utils.data import Dataset
import torch
from PIL import Image
from natsort import natsorted
import numpy as np
import PIL
from torchvision.transforms.functional import to_tensor


class CustomDataset(Dataset):
    def __init__(self, image_paths, target_paths, input_size):

        self.imgDir = image_paths
        self.maskDir = target_paths
        self.input_size = input_size
        self.Images = natsorted(os.listdir(self.imgDir))
        self.Masks = natsorted(os.listdir(self.maskDir))
        self.to_gray = transforms.Grayscale()

        # you should mapping for each class in your targets
        self.mapping = {0: 0, 255: 1}

    def mask_to_class(self, mask):
        for k in self.mapping:
            mask[mask == k] = self.mapping[k]
        return mask

    def _resize(self, image, mask):
        image = image.resize((self.input_size[0], self.input_size[1]), resample=PIL.Image.NEAREST)
        mask = mask.resize((self.input_size[0], self.input_size[1]), resample=PIL.Image.NEAREST)
        return image, mask

    def __getitem__(self, index):

        image = Image.open(self.imgDir + self.Images[index])
        mask = Image.open(self.maskDir + self.Masks[index])

        image, mask = self._resize(image, mask)

        image, mask = self.to_gray(image), self.to_gray(mask)

        image = np.array(image)
        tensor_image = to_tensor(image)

        mask = np.array(mask)
        mask = np.where(mask >= 250, 255, 0)
        mask = self.mask_to_class(mask)
        mask = torch.from_numpy(np.array(mask))
        mask = mask.long()

        return tensor_image, mask

    def __len__(self):  # return count of sample we have
        return len(self.Images)


# dataset = CustomDataset(image_paths="./DATA/deneme/imgs/",target_paths="./DATA/deneme/masks/")

# img,target = dataset[0]
# print(np.unique(target))
