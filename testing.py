from sklearn.cluster import KMeans
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

face= cascade.detectMultiScale(
                input,
                scaleFactor=1.1,
                minNeighbors=2,
                minSize=(50,50)
            )


window = face[0]
features = Feature_extractor(input)
pt = features.keypoints()
kp = features.select_kp(window)
if len(kp)==0:
    print "failed to find any keypoints"
    sys.exit(0) # just escape if it can't find anything

print "ANALYZING FEATURES"
Kp = np.asarray(pt)
segments = separate(Kp,50,50)
ratios = ratio_face(segments)


print "CLEARING THE DATA"
R = ratios[10:40]

print  "LOADING THE TRAINED DATA"

model = pickle.load(open("TRAINED_DECISION_TREE.pkl",'rb'))

print model.predict(R)






