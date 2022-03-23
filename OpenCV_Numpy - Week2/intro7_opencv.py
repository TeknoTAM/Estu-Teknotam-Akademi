"""

Why we use hsv color space?

"""

import cv2
import numpy as np

img = cv2.imread("./images/red_car.jpg")
hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

lower1 = np.array([161, 155, 84])
upper1 = np.array([179, 255, 255])

lower2 = np.array([0,50,50])
upper2 = np.array([10,255,255])

red_mask1 = cv2.inRange(hsv_img, lower1, upper1)
red_mask2 = cv2.inRange(hsv_img,lower2,upper2)

red_mask = red_mask1 + red_mask2
red = cv2.bitwise_and(img, img, mask=red_mask)

cv2.imshow("red_mask",red_mask)
cv2.imshow("red",red)
cv2.imshow("bgr",img)
cv2.imshow("hsv",hsv_img)
cv2.waitKey(0)