"""

Piksel değerlerini manipule etme    

"""
import cv2
import numpy as np


img = np.zeros(shape=(500,500,3),dtype=np.uint8)

cv2.imshow("img",img)
cv2.waitKey(0)

img[50,50] = (255,0,0) #just change blue channel
img[450,250] = (255,255,255)

cv2.imshow("img",img)
cv2.waitKey(0)

img_grayscale = np.full(shape=(500,500),fill_value=255,dtype=np.uint8)
img_grayscale[50,50] = 0


cv2.imshow("Grayscale",img_grayscale)
cv2.waitKey(0)


