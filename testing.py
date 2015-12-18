from sklearn.cluster import KMeans
from sklearn import tree
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
    distY = cluster(segmented['bottom'],18)[:,1]-cluster(segmented['left_eye'],1)[0][1]
    distX = abs(cluster(segmented['right_eye'],18)[:,0]-cluster(segmented['left_eye'],1)[0][0])
    A = distX/distY
    A.sort()
    return  zip([i for i in range(len(A))],A)



print "READING THE IMAGE FILE AND EXTRACTING KEYPOINTS"
input = cv2.imread("aron.jpg",0)
height = np.size(input,0)
width = np.size(input,1)
print height,width
cascade = cv2.CascadeClassifier('cascades/frontalface.xml')
model_filename = "TRAINED_SUPPORT_VECTOR_MACHINE.pkl"
model = pickle.load(open(model_filename,'rb'))



features = Feature_extractor(input)
pt = features.keypoints()

normalized=[(100*(x)/width,100*(y)/height) for (x,y) in pt]
pt=[(int(x),int(y)) for (x,y) in normalized]
print len(pt)

print "ANALYZING FEATURES"
Kp = np.asarray(pt)
segments = separate(Kp,50,50)
ratios = ratio_face(segments)

c ={1:'r',0:'b'}
X1=np.linspace(0,45,40)
Y1=np.linspace(0.5,1.5,40)
for x in X1:
    for y in Y1:
        label = model.predict([x,y])
        plt.scatter(x,y,color=c[int(label)])


for (x,y) in ratios:
    plt.scatter(x,y,color='black')



plt.show()








