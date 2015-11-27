
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
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








print "loading keypoints from pickle file"
keypoints_caprio = pickle.load(open("caprio_keypoints.pkl","rb"))
keypoints_aron = pickle.load(open("aron_keypoints.pkl","rb"))
c = {0:'r',1:'b',2:'k'}
pca = PCA(n_components=2)
clf = SVC()

C = np.asarray(keypoints_aron.features_())
A = np.asarray(keypoints_caprio.features_())



C_eyes = dist_eyes(C)[0:200]
A_eyes = dist_eyes(A)[0:200]


dist_eyes_mouth_C = dist_eyes_mouth(C)[0:200]
dist_eyes_mouth_A = dist_eyes_mouth(A)[0:200]

X = np.concatenate((zip(C_eyes,dist_eyes_mouth_C),zip(A_eyes,dist_eyes_mouth_A)),axis=0)

y = np.concatenate((np.ones(dist_eyes_mouth_C.shape[0]),np.zeros(dist_eyes_mouth_A.shape[0])),axis=0)

clf.fit(X,y)

pickle.dump(clf,open("trained_model.pkl",'wb'))




for ((x,y),(x1,y1)) in zip(zip(dist_eyes_mouth_C,C_eyes),zip(dist_eyes_mouth_A,A_eyes)):
    plt.scatter(x,y,color = 'r')
    plt.scatter(x1,y1)
    plt.xlabel("distance eyes to mouth")
    plt.ylabel("dis between eyes")




plt.show()
