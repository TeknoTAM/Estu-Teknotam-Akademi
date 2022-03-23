import cv2
import numpy as np


## Resim okuma
image = cv2.imread("./images/lena.jpeg")

## Resim özellikleri
print("Type of image: ",type(image))
print("Data type of image: ",image.dtype) # neden uint8 -> https://docs.opencv.org/2.4/modules/highgui/doc/user_interface.html#imshow


print(image.shape)
print(image.shape[1])
print(image.shape[2])
print(image.shape[3])


## Kanallara erişme
b,g,r = cv2.split(image) 

b = image[:,:,0]
g = image[:,:,1]
r = image[:,:,2]


## grayscale dönüştürme
print("Original image shape: ",image.shape)
gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
print("Grayscale image shape: ",gray_image.shape)
gray_image = cv2.imread("./images/rgb_image.jpeg",cv2.IMREAD_GRAYSCALE)


# resim görselleştirme
cv2.imshow("blue channel",b)
cv2.imshow("image",image)
cv2.imshow("gray image",gray_image)
cv2.waitKey(0)

