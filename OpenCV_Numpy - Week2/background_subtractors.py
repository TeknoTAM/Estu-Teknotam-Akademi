"""

If you use background subtractor, you must install opencv-contrib-python package

"""


import cv2
import numpy as np


# creating object
fgbg1 = cv2.bgsegm.createBackgroundSubtractorMOG()   
fgbg2 = cv2.createBackgroundSubtractorMOG2()
fgbg3 = cv2.bgsegm.createBackgroundSubtractorGMG()
  
cap = cv2.VideoCapture(0)
while(1):
    ret, img = cap.read()
      
    # apply mask for background subtraction
    fgmask1 = fgbg1.apply(img)
    fgmask2 = fgbg2.apply(img)
    print(fgbg2.getHistory())
    fgmask3 = fgbg3.apply(img)
      
    cv2.imshow('Original', img)
    cv2.imshow('GMG', fgmask3)
    cv2.imshow('MOG', fgmask1)
    cv2.imshow('MOG2', fgmask2)
    k = cv2.waitKey(30)
    if k == ord('q'):
        break
  
cap.release()
cv2.destroyAllWindows()