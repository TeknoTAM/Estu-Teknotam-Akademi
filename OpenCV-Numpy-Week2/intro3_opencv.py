"""

Opencv ile video okuma    

"""

import cv2
import numpy as np

cap = cv2.VideoCapture("./images/spot.mp4")

while True:
    ret,frame = cap.read()

    cv2.imshow("video",frame)
    key = cv2.waitKey(30)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()