from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn import tree
import numpy as np
import cv2
import matplotlib.pyplot as plt
from orb import Feature_extractor

__author__ = 'Abdullah_Rahman'

def normalization(input):
    X,Y = np.amax(input,axis=0)
    normalized=np.asarray([[100*(x)/X,100*(y)/Y] for (x,y) in input])
    return normalized

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
    distY = segmented['bottom'][:,1]-cluster(segmented['left_eye'],1)[0][1]
    distX = abs(segmented['right_eye'][:,0]-cluster(segmented['left_eye'],1)[0][1])


    if(len(distX)<len(distY)): A = distX/distY[0:len(distX)]
    if(len(distY)<len(distX)): A = distX[0:len(distY)]/distY
    A.sort()
    return  np.asarray(zip([i for i in range(len(A))],A))





print "READING THE IMAGE FILE "
aron = cv2.imread("aron.jpg",0)
caprio = cv2.imread("2.jpg",0)

test = cv2.imread("1.jpg",0)

print "EXTRACTING FEATURES"
aron_pt = normalization(Feature_extractor(aron).keypoints())
caprio_pt = normalization(Feature_extractor(caprio).keypoints())
test_pt = normalization(Feature_extractor(test).keypoints())



print len(aron_pt),len(caprio_pt)
print "PREPARING DATASET"
seg_1 = separate(aron_pt,50,50)
seg_2 = separate(caprio_pt,70,70)
seg_3 = separate(test_pt,70,65)

r1 = ratio_face(seg_1)
r2 = ratio_face(seg_2)
r3 = ratio_face(seg_3)

print "DEVELOPING DATASETS"
dat = np.append(r1,r2,axis=0)
label = np.append(np.zeros(len(r1)),np.ones(len(r2)))


print "STARTING MACHINE LEARNING "
clf = tree.DecisionTreeClassifier()
clf.fit(dat,label)



print "VIZ"
c = {0:'r',1:'b'}  # aron-->0 caprio---> 1
color_label = [c[i] for i in label]

#plt.scatter(dat[:,0],dat[:,1],color=color_label)
plt.scatter(dat[:,0],dat[:,1],color=color_label)
#plt.scatter(r2[:,0],r2[:,1],color='b')
res = clf.predict(r3)

print "ARON",res.tolist().count(0)
print "CAPRIO",res.tolist().count(1)
print len(res.tolist())

X1=np.linspace(0,100,40)
Y1=np.linspace(0,0.3,40)
for x in X1:
    for y in Y1:
        label = clf.predict([x,y])
        plt.scatter(x,y,color=c[int(label)])


#plt.show()


