"""

numpy.copy()   

"""
import numpy as np


if __name__ == "__main__":

    # assign array to another array
    arr1 = np.array([1,2])
    arr2 = arr1
    print("Arr2: ",arr2)
    arr1[0] = 3
    print("Arr2 after changing arr1:",arr2)


    # copy array to another array with numpy.copy
    arr1 = np.array([1,2])
    arr2 = arr1.copy()
    print("Arr2: ",arr2)
    arr1[0] = 3
    print("Arr2 after changing arr1: ",arr2)
    print(np.may_share_memory(arr1,arr2))

