import cv2
from sklearn.cluster import KMeans


__author__ = 'Abdullah_Rahman'

#(226.0, 42.0),(90.0, 254.0)

class Feature_extractor:
    def __init__(self,img):
        orb=cv2.ORB_create()
        self.kp=orb.detect(img)



    def keypoints(self):
        points=[]
        KP=self.kp

        for p in KP:
            points.append(p.pt)

        self.points=points
        return points

    def select_kp(self,window):
        (X,Y,W,H)=window
        selected=[]
        points=self.points
        selected=[(x,y) for (x,y) in points if ((x>X) & (x<(X+W)) & (y>Y )& (y<(Y+H))) ]
        normalized=[(100*(x-X)/W,100*(y-Y)/H) for (x,y) in selected]
        normalized_int=[(int(x),int(y)) for (x,y) in normalized]
        return normalized_int




def main():
    c=KMeans(n_clusters=3)
    img=cv2.imread("caprio.jpg",0)

    features=Feature_extractor(img)
    print features.keypoints()
    window=(140,132,200,200)

    features.keypoints()

    X=features.select_kp(window)
    for (x,y) in X:
        cv2.rectangle(img,(int(x),int(y)),(int(x)+3,int(y)+3),(0,255,0),1)
        print x,y



    cv2.imshow("test",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#main()