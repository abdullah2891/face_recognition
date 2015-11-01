from file_parser import File_parser
import os
import cv2

__author__ = 'Abdullah_Rahman'


absDir=os.path.dirname(__file__)
Images=[]


filename="raw/directories.txt"
fd=File_parser(filename)

for  s in fd.extracting():
    print s
    fileDir=absDir+"/"+s
    print fileDir
    img= cv2.imread(fileDir,0)
    Images.append([img])



print len(Images)




