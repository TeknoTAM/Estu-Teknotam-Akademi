import cv2
import numpy as np

# https://docs.opencv.org/3.4/db/d8e/tutorial_threshold.html


img = cv2.imread("./images/coins.jpg")
cv2.imshow("img",img)

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow("gray",gray)


# simple thresholding
threshold,thresh = cv2.threshold(gray,150,255,cv2.THRESH_BINARY) 
threshold,thresh_inv = cv2.threshold(gray,150,255,cv2.THRESH_BINARY_INV)

cv2.imshow("thresh invert",thresh_inv)
cv2.imshow("thresh",thresh)
cv2.waitKey(0)