import os
import torch
import torchvision
import cv2
import numpy as np
from PIL import Image
from torchsummary import summary

def get_model(feature_extract,pretrained,num_classes):

    model = torchvision.models.resnet50(pretrained=pretrained)
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False

    num_filters = model.fc.in_features
    model.fc = torch.nn.Linear(in_features= num_filters, out_features = num_classes)
    return model

def predict_multiclass(model,img):
    img = img.unsqueeze(0).cuda()
    logit = model(img)
    print("Logit: ",logit)
    probability = torch.softmax(logit,dim=1)[0].cpu().detach().numpy()
    print("Probability: ",probability)
    prediction = logit.argmax().cpu().numpy()
    print("Prediction: ",prediction)
    return prediction,probability,logit


if __name__ == "__main__":

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    test_data_path = "./data/test/"
    model_path = "./model.pth"

    model = get_model(feature_extract=False,pretrained=True,num_classes=3).to(device)
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    data_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size = (100,100)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) # (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    ])

    test_images = os.listdir(test_data_path)
    for index,image in enumerate(test_images):
        img = Image.open(test_data_path + image ).convert('RGB')
        vis_image = np.array(img)
        vis_image = cv2.cvtColor(vis_image,cv2.COLOR_RGB2BGR)

        img = data_transforms(img)
        img.to(device)

        with torch.no_grad():
            prediction,probability,logit = predict_multiclass(model,img)            
            cv2.putText(vis_image,str(prediction),(10,10),cv2.FONT_HERSHEY_PLAIN,1,(0,0,255),1)

        cv2.imshow("img",vis_image)
        key = cv2.waitKey(0)
        if key == ord('q'):
            break
        elif key == ord('d'):
            continue