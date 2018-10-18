import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches

#my imports 
import cv2
from LucasKanade import LucasKanade


# write your script here, we recommend the above libraries for making your animation
frames = np.load('../data/carseq.npy')

H, W, T = frames.shape
rect = np.array([59, 116, 145, 151]).T
p0 =  np.zeros(2)

for i in range(1, T):

	p = LucasKanade(frames[:, :, i-1], frames[:, :, i], rect, p0=p0)
	frame = frames[:, :, i].copy()
	print('p: ', p)
	rect[0] = rect[0] + p[1]
	rect[1] = rect[1] + p[0]
	rect[2] = rect[2] + p[1]
	rect[3] = rect[3] + p[0]
	print('rect: ', rect)
	

	cv2.rectangle(frame, (int(rect[0]),int(rect[1])), (int(rect[2]),int(rect[3])),(0, 0, 255), 3)
	if i in [2, 100, 200, 300, 400]:
		cv2.imwrite('frame_'+str(i) + '.jpg', frame.astype('uint8'))

	#break
	cv2.imshow('input', frame)
	print('frame: ', i)
	if cv2.waitKey(10) == ord('q'):
		break

cv2.destroyAllWindows()

