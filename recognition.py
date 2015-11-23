import pickle
import numpy as np

__author__ = 'Abdullah_Rahman'



keypoints_caprio = pickle.load(open("caprio_keypoints.pkl","rb"))
keypoints_aron = pickle.load(open("aron_keypoints.pkl","rb"))

C = np.asarray(keypoints_aron.features_())
A = np.asarray(keypoints_caprio.features_())

print C