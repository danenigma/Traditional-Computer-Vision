import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches

# write your script here, we recommend the above libraries for making your animation
import cv2
from LucasKanade import LucasKanade


# write your script here, we recommend the above libraries for making your animation
frames = np.load('../data/carseq.npy')

H, W, T = frames.shape
p0   =  np.zeros(2)

T_1  = frames[:, :, 0]
T_n = frames[:, :, 0]

rect0 = np.array([59, 116, 145, 151]).T
rect  = rect0.copy()
rect_n = rect0.copy()
p_n = p0.copy()

th  = 2.75

for i in range(1, T):

	I_n = frames[:, :, i]
	T_n = frames[:, :, i-1]
	
	p_n = LucasKanade(T_n, I_n, rect_n)
	
	if np.linalg.norm(p_n)>= th:
		p_n = LucasKanade(T_1, I_n, rect0)
	
	frame = frames[:, :, i].copy()
	#print('p: ', p)
	rect_n[0] = rect_n[0] + p_n[1]
	rect_n[1] = rect_n[1] + p_n[0]
	rect_n[2] = rect_n[2] + p_n[1]
	rect_n[3] = rect_n[3] + p_n[0]

	#print('rect: ', rect)
	
	#p_n_star = LucasKanade(T_1, I_n, rect_n, p0=p0)
	
	#rect[0] = rect[0] + p_n_star[1]
	#rect[1] = rect[1] + p_n_star[0]
	#rect[2] = rect[2] + p_n_star[1]
	#rect[3] = rect[3] + p_n_star[0]
	
	#print(p_n_star, p_n, np.linalg.norm(p_n_star - p_n))
	

	#if np.linalg.norm(p_n_star - p_n) < th:
	#	T_n = frames[:, :, i]						
	#	rect_n = rect.copy()
	#else:
	#	T_n = frames[:, :, i]
	#	print('I am here')

	cv2.rectangle(frame, (int(rect[0]),int(rect[1])), (int(rect[2]),int(rect[3])),(0, 0, 255), 3)
	if i in [2, 100, 200, 300, 400]:
		cv2.imwrite('frame_'+str(i) + '.jpg', frame.astype('uint8'))

	#break
	cv2.imshow('input', frame)
	print('frame: ', i)
	if cv2.waitKey(10) == ord('q'):
		break

cv2.destroyAllWindows()

