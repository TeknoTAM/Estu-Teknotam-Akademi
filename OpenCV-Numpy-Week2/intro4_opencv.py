import cv2
import numpy as np

"""
Imgeyi yeniden boyutlandirma

OpenCV'deki interpolasyon çeşitleri:
- INTER_NEAREST - a nearest-neighbor interpolation
- INTER_LINEAR - a bilinear interpolation (used by default)
- INTER_AREA - resampling using pixel area relation. It may be a preferred method for image decimation. But when the image is zoomed, it is similar to the INTER_NEAREST method.
- INTER_CUBIC - a bicubic interpolation over 4x4 pixel neighborhood
- INTER_LANCZOS4 - a Lanczos interpolation over 8x8 pixel neighborhood

"""

img = cv2.imread("./images/lena.jpeg")
print("Original image shape: ",img.shape)

new_img = cv2.resize(img,dsize=(200,200),interpolation=cv2.INTER_LINEAR) # interpolation
print("Shape after resizing: ",img.shape)

cv2.imshow("Img",img)
cv2.imshow("New img",new_img)
cv2.waitKey(0)