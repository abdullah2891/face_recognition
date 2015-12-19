from sklearn.cluster import KMeans
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

test = cv2.imread("5.jpg",0)

print "EXTRACTING FEATURES"
aron_pt = normalization(Feature_extractor(aron).keypoints())
caprio_pt = normalization(Feature_extractor(caprio).keypoints())

print len(aron_pt),len(caprio_pt)

print "PREPARING DATASET"
r1 = ratio_face(separate(aron_pt,50,50))
r2 = ratio_face(separate(caprio_pt,70,70))

print "DEVELOPING DATASETS"
dat = np.append(r1,r2,axis=0)
label = np.append(np.zeros(len(r1)),np.ones(len(r2)))


print "STARTING MACHINE LEARNING "
clf = tree.DecisionTreeClassifier()
clf.fit(dat,label)



print "VISUALIZING "
c = {0:'r',1:'b'}  # aron-->0 caprio---> 1
color_label = [c[i] for i in label]

#plt.scatter(dat[:,0],dat[:,1],color=color_label)
plt.scatter(dat[:,0],dat[:,1],color=color_label)
#plt.scatter(r2[:,0],r2[:,1],color='b')

print "TESTING "

filedir = ('1.jpg','2.jpg','3.jpg','4.jpg','5.jpg','6.jpg')

clf = KMeans(n_clusters=3)

for img in filedir:
    test = cv2.imread(img,0)
    test_pt = normalization(Feature_extractor(test).keypoints())
    cent = np.sort(clf.fit(test_pt).cluster_centers_,axis=0)
    midX =  (cent[:,0][0]+cent[:,0][1])/2
    midY =  (cent[:,1][0]+cent[:,1][2])/2
    r3 = ratio_face(separate(test_pt,midX,midY))
    res = clf.predict(r3)
    print "FIE ",img," ARON: ",float(res.tolist().count(0))/float(len(res.tolist()))," || CAPRIO: ",float(res.tolist().count(1))/float(len(res.tolist()))
    #print len(res.tolist())







