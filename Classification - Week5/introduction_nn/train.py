import torch
from tqdm import tqdm

from custom_model import Net
from dataset import CustomDataset


if __name__ == "__main__":
    batch_size = 1
    num_epochs = 120
    num_classes = 2
    lr = 1e-4
    
    model  = Net(num_classes=2)
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataset = CustomDataset(data_path="data/train/")
    train_loader = torch.utils.data.DataLoader(dataset,batch_size = batch_size,shuffle = True)

    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience = 2, factor = 0.05, verbose = True)
    
    criterion = torch.nn.BCEWithLogitsLoss() # for binary classification
    #criterion = torch.nn.BCELoss() # for binary classification with sigmoid layer

    for epoch in range(num_epochs):
        loop = tqdm(enumerate(train_loader),total=len(train_loader))
        running_loss = []

        for batch_index,(img,target) in loop:
            img,target = img.to(device),target.to(device)            
            
            output = model(img)
            target = target.unsqueeze(1).type(torch.float32) # for binary classification
            loss = criterion(output,target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss.append(loss.item())
            mean_loss = sum(running_loss) / len(running_loss)

            loop.set_description(f"Epoch: [{epoch + 1}/{num_epochs}]")
            loop.set_postfix(batch_loss = loss.item(), mean_loss = mean_loss, lr = optimizer.param_groups[0]["lr"]) 
            

        torch.save(model.state_dict(), "./binary_model.pth")
        scheduler.step(mean_loss)


