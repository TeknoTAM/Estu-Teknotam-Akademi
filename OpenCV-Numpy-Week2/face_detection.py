"""

Opencv cascade classifier ile yüz tespiti
Mantığını merak edenler için: https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html    

Farklı cascade türleri denemek isteyenler için diğer xml dosyalarının olduğu openCV reposu:
https://github.com/opencv/opencv/tree/master/data/haarcascades

"""

import cv2

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

image = cv2.imread("./images/lena.jpeg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=1,
    minSize=(2,2),
    flags = cv2.CASCADE_SCALE_IMAGE
)

for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("Faces Detected", image)
cv2.waitKey(0)