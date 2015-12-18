import pickle
import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn import tree

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
    distY = cluster(segmented['bottom'],100)[:,1]-cluster(segmented['left_eye'],1)[0][1]
    distX = abs(cluster(segmented['right_eye'],100)[:,0]-cluster(segmented['left_eye'],1)[0][0])
    A = distY/distX
    hist = np.histogram(A,bins=np.linspace(0,1,50))

    return  hist


print "LOADING KEYPOINTS FILES....."
caprio = pickle.load(open("caprio.pkl",'rb'))
aron = pickle.load(open('aron.pkl','rb'))

test_obj = pickle.load(open('test_keypoints.pkl','rb'))


feature_caprio = np.asarray(caprio.features_())
feature_aron = np.asarray(aron.features_())

test_data = np.asarray(test_obj.features_())

print 'SEGMENTING THE FACE'
segmented_face1 = separate(feature_caprio,y_threshold=50,x_threshold=50)
segmented_face2 = separate(feature_aron,y_threshold=50,x_threshold=50)

test =separate(test_data,y_threshold=50,x_threshold=50)

print "FINDING RATIO OF MOUTH AND EYES"
A = ratio_face(segmented_face1)
B= ratio_face(segmented_face2)

freq_A = A[0];freq_B = B[0]
intervals = A[1][0:49]

print "PREPARING DATASETS"
label =np.append(np.zeros(len(freq_A)),np.ones(len(freq_B)))
data = np.append(zip(intervals,freq_A),zip(intervals,freq_B),axis=0)



print "MODELING "
clf = tree.DecisionTreeClassifier()
clf.fit(data,label)

pickle.dump(clf,open("trained.pkl",'wb'))

print clf.predict(zip(intervals,freq_A))

plt.scatter(intervals,A[0])
plt.scatter(intervals,B[0],color='r')
plt.show()