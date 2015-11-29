import cv2
from sklearn.cluster import KMeans
from face_recog import Training
from orb import Feature_extractor
from file_parser import File_parser
import numpy as np
import sys

__author__ = 'Abdullah_Rahman'


#r=kath
threshold=0.53



filenames=("caprio.jpg","aron.jpg")
color=('r','b')
cascade="cascades/frontalface.xml"
C=KMeans(n_clusters=4)
face_cascade=cv2.CascadeClassifier(cascade)

image=cv2.imread("test.jpg",0)
features=Feature_extractor(image)

if image is None:
    print "IMAGE NOT FOUND in input_image.py"
    sys.exit(0)



face=face_cascade.detectMultiScale(
    image,
    scaleFactor=1.1,
    minNeighbors=2,
    minSize=(400,400)
)

window=face[0]
print face
(x,y,w,h)=window

print window

kp=features.keypoints()
print kp
selectKp=features.select_kp(window)

if len(selectKp)==0 :
    print "NO KEYPOINTS IN WINDOW"
    sys.exit(0)



clf=Training(filenames,color,cascade,READ_FILE=True)

clf.train()
points= clf.predict(selectKp)


print "counting frequency"

l= list(points)

first_image=list(l).count('r')
second_image=list(l).count('b')
first_percentage= float(first_image)/float(first_image+second_image)
second_percentage=float(second_image)/float(first_image+second_image)


print "confidence of first image/confidence of Second Image"
print first_percentage,second_percentage
if first_percentage>threshold:
    print filenames[0]+" IS THAT YOU!?"

if second_percentage>threshold:
    print filenames[1]+",Hi"




