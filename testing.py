from sklearn.cluster import KMeans
from sklearn import svm
import numpy as  np
from orb import Feature_extractor
import cv2
import sys
import matplotlib.pyplot as plt
import pickle
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
    distY = cluster(segmented['bottom'],40)[:,1]-cluster(segmented['left_eye'],1)[0][1]
    distX = abs(cluster(segmented['right_eye'],40)[:,0]-cluster(segmented['left_eye'],1)[0][0])
    A = distX/distY
    A.sort()
    return  zip([i for i in range(len(A))],A)



print "READING THE IMAGE FILE AND EXTRACTING KEYPOINTS"
input = cv2.imread("aron.jpg",0)
cascade = cv2.CascadeClassifier('cascades/frontalface.xml')
model_filename = "TRAINED_SUPPORT_VECTOR_MACHINE.pkl"
model = pickle.load(open(model_filename,'rb'))



features = Feature_extractor(input)
pt = features.keypoints()


print "ANALYZING FEATURES"
Kp = np.asarray(pt)
segments = separate(Kp,1200,900)
ratios = ratio_face(segments)

print model.predict(Kp)






