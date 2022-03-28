"""

Unsupervised Learning    
    
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


x1 = np.array([2,2,8,5,7,6,1,4])
x2 = np.array([10,5,4,8,5,4,2,9])

# show data with scatter plot
# plt.scatter(x1,x2)
# plt.xlabel("x1")
# plt.ylabel("x2")
# plt.show()

x1 = np.expand_dims(x1,axis=1)
x2 = np.expand_dims(x2,axis=1)
X = np.concatenate((x1,x2),axis=1)


Kmean = KMeans(n_clusters=3,)
Kmean.fit(X)
centroids = Kmean.cluster_centers_
print(Kmean.cluster_centers_)

plt.scatter(X[ : , 0], X[ : , 1], s =50, c="b")
plt.scatter(centroids[0][0], centroids[0][1], s=200, c='g', marker='s')
plt.scatter(centroids[1][0], centroids[1][0], s=200, c='r', marker='s')
plt.scatter(centroids[2][0], centroids[2][1], s=200, c='y', marker='s')
plt.show()