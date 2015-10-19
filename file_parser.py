import cv2
import os
__author__ = 'Abdullah_Rahman'


class File_parser:
    def __init__(self,filename):
        dir=os.path.dirname(__file__)
        self.absolute_dir=os.path.join(dir,filename)


    def extracting(self):
        file=open(self.absolute_dir,'r')
        l= file.read().split()
        self.l=l
        return l
    def label(self,color):
        L=[]
        for l in self.l:
            L.append(color)

        return L


def main():
    fd=File_parser('raw/directories.txt')
    fd.extracting()
    print fd.label('b')+fd.label('r')

#main()