"""

Why we should numpy?    

"""


import numpy as np
import cv2
import time

# our function to change pixels that equal to 150
def change_pixelValue(arr):
    for x in range(arr.shape[0]):
        for y in range(arr.shape[1]):
            if arr[y,x] == 150:
                arr[y,x] = 255
    return arr

if __name__ == "__main__":

    arr = np.ones((200,200),dtype=np.uint8)
    arr[50:100,50:100] = 150 #change pixel's value in spesific area

    cv2.imshow("arr",arr)
    cv2.waitKey(0)


    since = time.time()
    arr = change_pixelValue(arr) # call our function 
    #arr = np.where(arr == 150,255,arr) # numpy method

    print("Total time: ",time.time() - since) # comment our function,uncomment np.where method, see time difference between two methods


    cv2.imshow("img",arr)
    cv2.waitKey(0)




