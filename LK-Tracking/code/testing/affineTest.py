import cv2
import numpy as np
from LucasKanadeAffine import LucasKanadeAffine
from scipy.ndimage import affine_transform
# write your script here, we recommend the above libraries for making your animation
frames = np.load('../data/aerialseq.npy')

H, W, T = frames.shape
M = np.array([[1, 0, 0],
			  [0, 1, 5]])


for i in range(1,2):
	
	#mask = SubtractDominantMotion(frames[:,:,i-1], frames[:,:,i])
	warp_frame = affine_transform(frames[:,:,i],  M[:2,:2], offset = [M[0, 2], M[1, 2]])
	M_est = LucasKanadeAffine(frames[:,:,i], warp_frame)
	M_temp = np.eye(3,3)
	M_temp[:2, :] = M_est
	M_est_inv = np.linalg.inv(M_temp)
	#M_est_inv = np.linalg.inv(M)
	#frame_est = M_est_inv @ 
	frame_est   = affine_transform(warp_frame,  M_est_inv[:2,:2], 
											offset =[M_est_inv[0, 2], M_est_inv[1, 2]])
	#print('M_est: ', M_est)
	
	cv2.imshow('input', frames[:,:,i])
	cv2.imshow('warp', warp_frame)
	cv2.imshow('recon', frame_est)
	
#	cv2.imshow('mask', mask)

	if cv2.waitKey(0) == ord('q'):
		break

cv2.destroyAllWindows()
