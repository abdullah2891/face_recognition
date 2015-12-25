from sklearn.cluster import KMeans
from sklearn import tree
import numpy as np
import cv2
import matplotlib.pyplot as plt
from orb import Feature_extractor
import os,sys
from file_parser import File_parser
from toyNN import synapse

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


print "EXTRACTING FEATURES"
aron_pt = normalization(Feature_extractor(aron).keypoints())
caprio_pt = normalization(Feature_extractor(caprio).keypoints())

#print len(aron_pt),len(caprio_pt)

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
absDir=os.path.dirname(__file__)
Images_dir=[]


directory_file = "directories.txt"
filename="testing/sample"
fd=File_parser(filename+'\\'+directory_file)

print "EXTRACTING IMAGES"
for  s in fd.extracting():
    fileDir=absDir+'/'+filename+"/"+s
    Images_dir.append(fileDir)
    #print fileDir

cl = KMeans(n_clusters=3)


print "File,ARON,CAPRIO"
for img in Images_dir:
    test = cv2.imread(img,0)
    #print img,len(test)
    test_pt = normalization(Feature_extractor(test).keypoints())
    cent = np.sort(cl.fit(test_pt).cluster_centers_,axis=0)
    midX =  (cent[:,0][1]+cent[:,0][2])/2
    midY =  (cent[:,1][0]+cent[:,1][2])/2
    plt.scatter(test_pt[:,0],test_pt[:,1])
    r3 = ratio_face(separate(test_pt,midX,midY))
    res = clf.predict(r3)
    print img[81:len(img)],",",float(res.tolist().count(0))\
                               /float(len(res.tolist())),",",\
        float(res.tolist().count(1))/float(len(res.tolist()))



#plt.show()




