import pickle
import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm import SVC
import matplotlib.pyplot as plt

__author__ = 'Abdullah_Rahman'
def separate(C,y_threshold,x_threshold):
    bottom =C[np.where(C[:,1]>y_threshold)]
    right = C[np.where((C[:,0]<x_threshold) & (C[:,1]<y_threshold))]
    left = C[np.where((C[:,0]>x_threshold) & (C[:,1]<y_threshold))]
    segmented = {'bottom': bottom, 'right_eye':right,'left_eye':left}

    return segmented

def cluster(A,number_cluster):
    C = KMeans(n_clusters=number_cluster)
    C.fit(A)
    return C.cluster_centers_


def ratio_face(segmented):
    distY = cluster(segmented['bottom'],300)[:,1]-cluster(segmented['left_eye'],1)[0][1]
    distX = abs(cluster(segmented['right_eye'],300)[:,0]-cluster(segmented['left_eye'],1)[0][0])
    A = distX/distY
    A.sort()
    return  zip([i for i in range(len(A))],A)

