import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from natsort import natsorted


class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root):
        self.root = root
        self.imgs = list(natsorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(natsorted(os.listdir(os.path.join(root, "PedMasks"))))

        self.transforms = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        mask = np.array(mask)
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]

        masks = mask == obj_ids[:, None, None]

        # get bbox
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a Torch tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        img = self.transforms(img)
        # target = self.transforms(target)

        # if self.transforms is not None:
        #     img,target = self.transforms(img,target)

        return img, target

    def __len__(self):
        return len(self.imgs)


# pseudo code
# if __name__ == "__main__":
#     Dataset = PennFudanDataset(root="./PennFudanPed")

#     train_loader = DataLoader(dataset=Dataset, batch_size=1)

#     for epoch in range(1):
#         for img, target in train_loader:
#             print("Image shape: ", img.shape)
#             print("Image type: ", type(img))
#             print("Image data type: ", img.dtype)
#             print("***********")
