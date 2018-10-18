import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
from SubtractDominantMotion import SubtractDominantMotion

import cv2
# write your script here, we recommend the above libraries for making your animation
frames = np.load('../data/aerialseq.npy')

H, W, T = frames.shape

for i in range(1,T):
	print('frame: ', i)
	mask = SubtractDominantMotion(frames[:,:,i-1], frames[:,:,i])
	cv2.imshow('input', frames[:,:,i])
	cv2.imshow('mask', mask)
	

	if cv2.waitKey(10) == ord('q'):
		break

cv2.destroyAllWindows()
