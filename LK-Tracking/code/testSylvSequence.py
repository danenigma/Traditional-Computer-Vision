import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
import cv2
# write your script here, we recommend the above libraries for making your animation
from LucasKanadeBasis import LucasKanadeBasis
from LucasKanade import LucasKanade



# write your script here, we recommend the above libraries for making your animation
frames = np.load('../data/sylvseq.npy')
bases = np.load('../data/sylvbases.npy')

H, W, T = frames.shape
sylvseqrects = np.zeros((T, 4))

rect1 = np.array([101, 61, 155, 107]).astype('float').T
rect2 = np.array([101, 61, 155, 107]).astype('float').T


for i in range(1, T):
	p1 = LucasKanadeBasis(frames[:, :, i-1], frames[:, :, i], rect1, bases)
	p2 = LucasKanade(frames[:, :, i-1], frames[:, :, i], rect2)
	rect1 += np.array([p1[1], p1[0], p1[1], p1[0]]).T	

	sylvseqrects[i, :] = rect1
	rect2 += np.array([p2[1], p2[0], p2[1], p2[0]]).T	

	print('frame: ', i)
	
	frame = frames[:, :, i].copy()
	cv2.rectangle(frame, (int(rect1[0]),int(rect1[1])), (int(rect1[2]),int(rect1[3])),(255), 1)
	cv2.rectangle(frame, (int(rect2[0]),int(rect2[1])), (int(rect2[2]),int(rect2[3])),(0), 1)

	cv2.imshow('input', frame)

	if cv2.waitKey(10) == ord('q'):
		break
np.save('sylvseqrects.npy', sylvseqrects)
cv2.destroyAllWindows()

