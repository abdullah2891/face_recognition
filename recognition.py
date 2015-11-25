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






print "loading keypoints from pickle file"
keypoints_caprio = pickle.load(open("caprio_keypoints.pkl","rb"))
keypoints_aron = pickle.load(open("aron_keypoints.pkl","rb"))
c = {0:'r',1:'b',2:'k'}

C = np.asarray(keypoints_aron.features_())
A = np.asarray(keypoints_caprio.features_())

caprio_cluster = clustering(C,3)
aron_cluster = clustering(A,3)

print caprio_cluster.labels_





for ((x,y),labels) in zip(C,caprio_cluster.labels_):
    plt.scatter(x,y,color = c[labels])


plt.show()

