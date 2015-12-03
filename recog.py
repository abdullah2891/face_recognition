import pickle
import numpy as np

__author__ = 'Abdullah_Rahman'

def separate(C,y_threshold,x_threshold):
    bottom =C[np.where(C[:,1]<y_threshold)]
    right = C[np.where((C[:,0]<x_threshold) & (C[:,1]>y_threshold))]
    left = C[np.where((C[:,0]>x_threshold) & (C[:,1]>y_threshold))]
    segmented = {'bottom': bottom, 'right_eye':right,'left_eye':left}

    return segmented




print "LOADING FILES....."
caprio = pickle.load(open("aron_keypoints.pkl",'rb'))
aron = pickle.load(open('caprio_keypoints.pkl','rb'))

feature_caprio = np.asarray(caprio.features_())
feature_aron = np.asarray(aron.features_())


print separate(feature_caprio,y_threshold=50,x_threshold=50)