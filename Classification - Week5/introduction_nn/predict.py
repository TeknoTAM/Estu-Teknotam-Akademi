import os
import torch
import torchvision
import numpy as np
import cv2
from PIL import Image

from custom_model import Net

def predict_binary(model,img):
    prediction = None
    img = img.unsqueeze(0).cuda()
    logit = model(img)    
    print("Raw logit: ",logit)
    probability = torch.sigmoid(logit).cpu().detach().numpy()
    print("Probability: ",probability[0][0])
    if probability[0][0] < 0.5:
        prediction = 0
    elif probability[0][0] >= 0.5:
        prediction = 1
    return prediction,probability,logit


if __name__ == "__main__":
    num_classes = 2
    input_size = (256,256)
    test_data_path = "./data/test/"

    transform = torchvision.transforms.Compose([torchvision.transforms.Resize(size = (256,256)),
                                                torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    model = Net(num_classes=num_classes)
    model.load_state_dict(torch.load("binary_model.pth"))
    model.to(device)

    test_images = os.listdir(test_data_path)
    for index,image_name in enumerate(test_images):
        img = Image.open(test_data_path + image_name)
        vis_img = np.array(img)
        vis_img = cv2.cvtColor(vis_img,cv2.COLOR_RGB2BGR)

        img = transform(img)
        img.to(device)
        with torch.no_grad():
            prediction,probability,logit = predict_binary(model,img) 
            cv2.putText(vis_img,str(prediction),(10,10),cv2.FONT_HERSHEY_PLAIN,1,(0,0,255),1)

        cv2.imshow("img",vis_img)
        key = cv2.waitKey(0)
        if key == ord('q'):
            break
        elif key == ord('d'):
            continue
        
        