import torch
import torchvision
import torchsummary
from tqdm import tqdm

def get_model(feature_extract,pretrained,num_classes):

    model = torchvision.models.resnet50(pretrained=pretrained)
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False

    num_filters = model.fc.in_features
    model.fc = torch.nn.Linear(in_features= num_filters, out_features = num_classes)
    return model



if __name__ == "__main__":

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    num_epochs = 10

    model = get_model(feature_extract=False,pretrained=True,num_classes=3)
    model.to(device)


    data_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size = (100,100)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) # (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    ])

    train_data = "./data/train/"
    trainset = torchvision.datasets.ImageFolder(train_data,transform=data_transforms)
    train_loader = torch.utils.data.DataLoader(dataset = trainset, batch_size = 2,shuffle = True)


    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience = 2, factor = 0.05, verbose = True)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        running_loss = []
        loop = tqdm(enumerate(train_loader),total = len(train_loader))
        for batch_index, (data,target) in loop:
            data,target = data.to(device), target.to(device)

            output = model(data)
            loss = criterion(output,target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss.append(loss.item())
            mean_loss = sum(running_loss) / len(running_loss)
            loop.set_description(f"Epoch: [{epoch + 1}/{num_epochs}]")
            loop.set_postfix(batch_loss = loss.item(), mean_loss = mean_loss, lr = optimizer.param_groups[0]["lr"]) 

        torch.save(model.state_dict(), "./model_multiclass.pth")
        scheduler.step(mean_loss)
