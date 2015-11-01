import cv2
from sklearn.cluster import KMeans
from face_recog import Training
from orb import Feature_extractor
from file_parser import File_parser
import numpy as np


__author__ = 'Abdullah_Rahman'

referenceImage=("caprio.jpg","aron.jpg")
label=('r','b')
cascade="frontalface.xml"


class Recognize:
    def __init__(self,referenceImage,label,cascade):
        self.clf=Training(referenceImage,label,cascade)
        self.clf.train()
        self.face_cascade=cv2.CascadeClassifier(cascade)

    def label(self,target):
        image=cv2.imread(target,0)
        features=Feature_extractor(image)

        face=self.face_cascade.detectMultiScale(
            image,
            scaleFactor=1.1,
            minNeighbors=2,
            minSize=(10,10)
        )

        if len(face)==0:
            return -1

        window=face[0]
        (x,y,w,h)=window

        kp=features.keypoints()
        selectKp=features.select_kp(window)

        labels=self.clf.predict(selectKp)
        return labels




def main():
    referenceImage=("caprio.jpg","test.jpg")
    label=('r','b')
    cascade="frontalface.xml"
    image=cv2.imread("caprio.jpg",0)
    r=Recognize(referenceImage,label,cascade)


    print r.label("caprio.jpg")

main()


