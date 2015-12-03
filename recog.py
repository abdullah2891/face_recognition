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


print "LOADING KEYPOINTS FILES....."
caprio = pickle.load(open("caprio_keypoints.pkl",'rb'))
aron = pickle.load(open('aron_keypoints.pkl','rb'))

feature_caprio = np.asarray(caprio.features_())
feature_aron = np.asarray(aron.features_())

print 'SEGMENTING THE FACE'
segmented_face1 = separate(feature_caprio,y_threshold=50,x_threshold=50)
segmented_face2 = separate(feature_aron,y_threshold=50,x_threshold=50)

print "FINDING RATIO OF MOUTH AND EYES"
A = ratio_face(segmented_face1)
B= ratio_face(segmented_face2)

print "APPLYING CUTOFF TO CLEANUP DATA"
A = A[50:250]
B = B[50:250]

print "STARTING MACHINE LEARNING"
clf = SVC()

label =np.append(np.ones(len(A)),np.zeros(len(B)))
data = np.append(A,B,axis=0)

clf.fit(data,label)

pickle.dump(clf,open("TRAINED_SVM.pkl",'wb'))


print "VISUALIZING "
c ={1:'r',0:'b'}
for ((x,y),l) in zip(data,label):
    plt.scatter(x,y,color=c[l])

plt.show()



