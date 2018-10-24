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
rect0 = np.array([59, 116, 145, 151]).astype('float').T
rect = rect0.copy()

p0 =  np.zeros(2).astype('float')
carseqrects = np.zeros((T, 4))
carseqrects[0, :] = rect
 

for i in range(1, T):
	
	rect_prev = rect.copy()
	
	p = LucasKanade(frames[:, :, i-1], frames[:, :, i], rect)
	
	rect += np.array([p[1], p[0], p[1], p[0]]).T
	
	p0 = np.array([rect[1]-rect0[1], rect[0]-rect0[0]]).T
		
	p_star = LucasKanade(frames[:, :, 0], frames[:, :, i], rect0, p0)

	p_star[1] = p_star[1] - (rect_prev[0] - rect0[0])
	p_star[0] = p_star[0] - (rect_prev[1] - rect0[1])
	
	if np.linalg.norm(p_star-p)<= 2.5:
		p = p_star

	rect = rect_prev + np.array([p[1], p[0], p[1], p[0]]).T
	
	carseqrects[i, :] = rect
	
	frame = frames[:, :, i].copy()
	cv2.rectangle(frame, (int(rect[0]),int(rect[1])), (int(rect[2]),int(rect[3])),(255), 1)
	if i in [2, 100, 200, 300, 400]:
		cv2.imwrite('frame_'+str(i) + '.jpg', frame.astype('uint8'))

	cv2.imshow('input', frame)
	print('frame: ', i)
	if cv2.waitKey(10) == ord('q'):
		break
np.save('carseqrects.npy', carseqrects)
cv2.destroyAllWindows()

