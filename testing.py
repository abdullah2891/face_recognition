import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pickle
from sklearn.svm import SVC

__author__ = 'Abdullah_Rahman'

def clustering(dataset,number_of_cluster):
    print "Clustering"
    clusterA = KMeans(n_clusters=number_of_cluster)
    clusterA.fit(dataset)
    return clusterA


def separate_data(C,cluster_object,label):    # not using this for now
    print "SEPERATING DATA BASED ON LABEL"
    label_ref = cluster_object.labels_
    pos = np.where(label_ref==label)
    print pos
    return C[pos]


def dist_eyes(C):
    right_c = C[np.where((C[:,1]<50) & (C[:,0]>40))]
    ref_C = clustering(C[np.where((C[:,1]<50) & (C[:,0]<40))],1).cluster_centers_
    dist_eyes_C = right_c[:,0]-ref_C[0][0]
    return dist_eyes_C

def dist_eyes_mouth(C):
    bottom_half_c = C[np.where(C[:,1]>50)]
    ref_C = clustering(C[np.where((C[:,1]<50) & (C[:,0]<40))],1).cluster_centers_
    dist_C = bottom_half_c[:,1]-ref_C[0][1]
    return dist_C


svm_model = pickle.load(open("trained_model.pkl","rb"))
keypoint = pickle.load(open("test_keypoints.pkl","rb"))

kp = np.asarray(keypoint.features_())

dist_eyes = dist_eyes(kp)[0:200]
dist_eyes_mouth = dist_eyes_mouth(kp)[0:200]

X= zip(dist_eyes,dist_eyes_mouth)

l = svm_model.predict(X).tolist()

print "caprio ",l.count(1),"aron ",l.count(0),len(l)