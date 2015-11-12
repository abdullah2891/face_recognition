from file_parser import File_parser
import os
import cv2
from Recognize import *

__author__ = 'Abdullah_Rahman'


absDir=os.path.dirname(__file__)
Images=[]

label=('r','b','r','b')
cascade="frontalface.xml"






filename="raw/directories.txt"
fd=File_parser(filename)

for  s in fd.extracting():
    fileDir=absDir+"/"+s
    Images.append(fileDir)


for image in Images:
    print image
    detector = Recognize(image,label,cascade)




