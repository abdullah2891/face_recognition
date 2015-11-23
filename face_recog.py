import cv2
from orb import Feature_extractor
from matplotlib import pyplot as plt
from sklearn import tree
import sys

__author__ = 'Abdullah_Rahman'

#"myfacedetector.xml"


class Training:

    READ_FILE = True
    def __init__(self,filename,color,cascade,READ_FILE):

        labeled=zip(filename,color)
        label=[]
        feature_train=[]

        for (fd,c) in labeled:
            if READ_FILE is True:
                img=cv2.imread(fd,0)
                img =cv2.blur(img,(5,5))
            else:
                img = filename
            if img is None:
                print "image can't be loaded in face recog"
                sys.exit(0)
            face_cascade=cv2.CascadeClassifier(cascade)
            features=Feature_extractor(img)

            face=face_cascade.detectMultiScale(
                img,
                scaleFactor=1.1,
                minNeighbors=2,
                minSize=(10,10)
            )

            window=face[0]
            (x,y,w,h)=window


            #print "selectkp"
            #print window

            kp=features.keypoints()
            selectKp=features.select_kp(window)
            #print selectKp


            for (X,Y) in selectKp:
                plt.scatter(X,Y,color=c)
                label.append(c)
                feature_train.append([X,Y])


            #cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)

            #cv2.imshow("image",img)


        self.label=label
        self.feature_train=feature_train
        #print label,feature_train
        #plt.show()

    def train(self):
        self.clf=tree.DecisionTreeClassifier()
        self.clf.fit(self.feature_train,self.label)
    def predict(self,input):
        return self.clf.predict(input)


    def features_(self):
        return self.feature_train

    def labels_(self):
        return self.label





def main():
    filenames=('a10.bmp','a4.bmp')
    color=('r','b')
    cascade="frontalface.xml"
    train=Training(filenames,color,cascade)



