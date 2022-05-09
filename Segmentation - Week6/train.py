# https://discuss.pytorch.org/t/error-in-python-s-multiprocessing-library/31355/43
# https://discuss.pytorch.org/t/pytorch-equivalence-to-sparse-softmax-cross-entropy-with-logits-in-tensorflow/18727/2
# https://discuss.pytorch.org/t/multiclass-segmentation/54065/5
# https://discuss.pytorch.org/t/training-semantic-segmentation/49275/4

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from models.model import UNet
from utils.dataset import CustomDataset

def train(epochs,trainLoader):
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience = 1, factor = 0.1,verbose=True)    
    
    mean_losses = []
    print("[INFO] Training is started.")
    for epoch in range(epochs):
        running_loss = []
        loop = tqdm(enumerate(trainLoader),total=len(trainLoader))

        for idx, (image,mask) in loop:
            image,mask = image.to(device),mask.to(device)
            outputs = model(image)
            loss = criterion(outputs,mask)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss.append(loss.item())
            mean_loss = sum(running_loss) / len(running_loss)
            
            loop.set_description(f"Epoch: [{epoch + 1}/{epochs}]")
            loop.set_postfix(batch_loss = loss.item(), mean_loss = mean_loss, lr = optimizer.param_groups[0]["lr"]) 

        if len(mean_losses) >= 1: 
            if mean_loss < min(mean_losses):
                print("[INFO] Model saved.")
                torch.save(model.state_dict(), "model_v2.pth") 


        mean_losses.append(mean_loss)
        scheduler.step(mean_loss)
        

if __name__=='__main__':
    
    img_paths = "./DATA/train/images/"
    mask_paths = "./DATA/train/masks/"

    train_dataset = CustomDataset(img_paths, mask_paths,input_size=(256,256))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=8)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = UNet(num_classes=2)
    model = model.to(device)
        
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    train(epochs=350,trainLoader=train_loader)
    print("[INFO] Training is ended.")
   
    