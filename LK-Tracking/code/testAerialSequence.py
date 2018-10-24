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

	mask = SubtractDominantMotion(frames[:,:,i-1], frames[:,:,i])
	mask_rgb = cv2.cvtColor((mask*255).astype('uint8'),cv2.COLOR_GRAY2RGB)
	mask_rgb[:, : , :2] = 0
	
	frame_rgb = cv2.cvtColor((frames[:,:,i].copy()*255).astype('uint8'), cv2.COLOR_GRAY2RGB)
	
	out = cv2.add(frame_rgb, mask_rgb)

	cv2.imshow('input', frames[:,:,i])
	cv2.imshow('mask', out)
	

	if cv2.waitKey(10) == ord('q'):
		break

cv2.destroyAllWindows()
