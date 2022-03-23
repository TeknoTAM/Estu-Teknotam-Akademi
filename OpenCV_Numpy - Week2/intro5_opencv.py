"""

Resim üzerine farkli şekiller ve yazı ekleme    

"""

import cv2
import numpy as np


blank = np.zeros((500,500,3),dtype=np.uint8)
cv2.imshow('Blank',blank)

# point image a certain color 
blank[0:100,0:100] = 0,255,0 # 100x100 lük bir alanı boya

# draw rectangle
cv2.rectangle(blank,(50,50),(100,100),(0,0,255),1) # trt with thickness=-1

# draw circle
cv2.circle(blank,(255,250),50,(0,0,255),1)

# draw line 
cv2.line(blank,(0,0),(300,300),(0,0,255),2)

# put text
cv2.putText(blank,'ESTU',(70,70),cv2.FONT_HERSHEY_PLAIN,1,(0,0,255),1)


cv2.imshow("Green",blank)
cv2.waitKey(0)