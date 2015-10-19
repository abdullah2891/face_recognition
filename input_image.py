import cv2
from sklearn.cluster import KMeans
from face_recog import Training
from orb import Feature_extractor
from file_parser import File_parser
import numpy as np

__author__ = 'Abdullah_Rahman'


#r=kath
threshold=0.55

filenames=("caprio.jpg","guy.jpg")
color=('r','b')
cascade="frontalface.xml"
C=KMeans(n_clusters=4)
face_cascade=cv2.CascadeClassifier(cascade)

image=cv2.imread("tes1.jpg",0)
features=Feature_extractor(image)

face=face_cascade.detectMultiScale(
    image,
    scaleFactor=1.1,
    minNeighbors=2,
    minSize=(50,50)
)

window=face[0]
(x,y,w,h)=window



kp=features.keypoints()

selectKp=features.select_kp(window)
            #print selectKp




clf=Training(filenames,color,cascade)

clf.train()

points= clf.predict(selectKp)


print "counting frequency"

l= list(points)

Kath=list(l).count('r')
Reich=list(l).count('b')
Kath_percentage= float(Kath)/float(Kath+Reich)
Reich_percentage=float(Reich)/float(Kath+Reich)

print Kath_percentage,Reich_percentage
if Kath_percentage>threshold:
    print filenames[0]+" IS THAT YOU!?"

if Reich_percentage>threshold:
    print filenames[1]+",Hi"



