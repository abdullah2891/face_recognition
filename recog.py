import pickle
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
__author__ = 'Abdullah_Rahman'

def separate(C,y_threshold,x_threshold):
    bottom =C[np.where(C[:,1]>y_threshold)]
    right = C[np.where((C[:,0]<x_threshold) & (C[:,1]<y_threshold))]
    left = C[np.where((C[:,0]>x_threshold) & (C[:,1]<y_threshold))]
    segmented = {'bottom': bottom, 'right_eye':right,'left_eye':left}

    return segmented

def clustering(A,number_cluster):
    C = KMeans(n_clusters=number_cluster)
    C.fit(A)
    return C

def distance(A,reference,axis):
    return




print "LOADING FILES....."
caprio = pickle.load(open("aron_keypoints.pkl",'rb'))
aron = pickle.load(open('caprio_keypoints.pkl','rb'))

feature_caprio = np.asarray(caprio.features_())
feature_aron = np.asarray(aron.features_())


segmented = separate(feature_caprio,y_threshold=50,x_threshold=50)


plt.scatter(segmented['left_eye'][:,0],segmented['left_eye'][:,1],color='g')
plt.scatter(segmented['bottom'][:,0],segmented['bottom'][:,1],color='r')
plt.scatter(segmented['right_eye'][:,0],segmented['right_eye'][:,1],color='b')

plt.show()