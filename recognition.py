import pickle
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

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




print "loading keypoints from pickle file"
keypoints_caprio = pickle.load(open("caprio_keypoints.pkl","rb"))
keypoints_aron = pickle.load(open("aron_keypoints.pkl","rb"))
c = {0:'r',1:'b',2:'k'}

C = np.asarray(keypoints_aron.features_())
A = np.asarray(keypoints_caprio.features_())

caprio_cluster = clustering(C,3)
aron_cluster = clustering(A,3)

upper_half_c = C[np.where(C[:,1]<50)]
upper_half_a = A[np.where(A[:,1]<50)]

bottom_half_c = C[np.where(C[:,1]>50)]
bottom_half_a = A[np.where(A[:,1]>50)]




for ((x,y),(x1,y1)) in zip(bottom_half_a,upper_half_a):
    plt.scatter(x,y)
    plt.scatter(x1,y1,color='r')


plt.show()

