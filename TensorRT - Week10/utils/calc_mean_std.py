import torch
import torchvision.transforms as transforms
import torchvision


# There is difference between last function in computing std.
# this function computes mean and std for each batch over pixels, take the average of them.
# this called Batch-Normalization
def calc_mean_std(train_data,input_size,batch_size):
    data_transforms = transforms.Compose([
    transforms.Resize(size = (input_size)),
    transforms.ToTensor(),
    ])

    trainset = torchvision.datasets.ImageFolder(train_data,transform=data_transforms)
    train_loader = torch.utils.data.DataLoader(dataset = trainset, batch_size = batch_size,shuffle = True)

    mean,std = 0., 0.
    nb_samples = 0
    for image,target in train_loader:
        # assume image shape is [1,3,100,100] --> (BATCH,CHANNEL,H,W) 
        batch_samples = image.size(0) # for how many images are there in one batch
        image = image.view(batch_samples,image.size(1),-1) # convert (1,3,10000)
        mean += image.mean(2).sum(0)
        std += image.std(2).sum(0)
        nb_samples += batch_samples
    mean /= nb_samples
    std /= nb_samples
    print(f"Mean: {mean}, Std: {std}")
    return mean,std


# that is not True, this is not Batch-Normalization
def calc_mean_std2(train_data,input_size,batch_size):
    data_transforms = transforms.Compose([
    transforms.Resize(size = (input_size)),
    transforms.ToTensor(),
    ])

    trainset = torchvision.datasets.ImageFolder(train_data,transform=data_transforms)
    train_loader = torch.utils.data.DataLoader(dataset = trainset, batch_size = batch_size,shuffle = True)

    mean,meansq = 0.,0.
    for image,target in train_loader:
        mean = image.mean()
        meansq = (image **2).mean()

    std = torch.sqrt(meansq - mean**2)
    print(f"Mean: {mean}, Std: {std}")
    return mean,std

# compute mean of the entire data and std, different from batch normalization
# std is too high, because calculate on entire data
def calc_mean_std3(train_data,input_size,batch_size):
    data_transforms = transforms.Compose([
    transforms.Resize(size = (input_size)),
    transforms.ToTensor(),
    ])

    trainset = torchvision.datasets.ImageFolder(train_data,transform=data_transforms)
    train_loader = torch.utils.data.DataLoader(dataset = trainset, batch_size = batch_size,shuffle = True)
    mean,std = 0.,0.
    for images,_ in train_loader:
        # assume images shape is [1,3,100,100] --> BATCH,CHANNEL,H,W
        batch_samples = images.size(0) # get 1
        images = images.view(batch_samples,images.size(1),-1) # convert (1,3,10000)
        mean += images.mean(2).sum(0) #add channel values 
    mean = mean / len(train_loader.dataset)

    var = 0.0
    for images,_ in train_loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples,images.size(1), -1)
        var += ((images - mean.unsqueeze(1)) **2).sum([0,2])
    std = torch.sqrt(var / len(train_loader.dataset)*224*224)

    print(f"Mean: {mean}, Std: {std}")
    return mean,std