import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import shift
import matplotlib.pyplot as plt
#my imports
import cv2

def LucasKanade_shift(It, It1, rect, p0 = np.zeros(2)):
	# Input: 
	#	It: template image
	#	It1: Current image
	#	rect: Current position of the car
	#	(top left, bot right coordinates)
	#	p0: Initial movement vector [dp_x0, dp_y0]
	# Output:
	#	p: movement vector [dp_x, dp_y]
	# Put your implementation here
	
	threshold = 0.00001
	p = p0
	
	It = np.float32(It)/np.max(It)
	It1 = np.float32(It1)/np.max(It1)
	
	
	YY, XX = np.meshgrid(np.arange(rect[0], rect[2]), np.arange(rect[1],rect[3]))
	warp_jacobian = np.array([[1, 0],[0, 1]]).astype('float')

	[It1_x, It1_y] = np.gradient(It1) 
		
	
	while True:	
			
		
		It1_w   = shift(It1,   (p[0], p[1])) 
		It1_x_w = shift(It1_x, (p[0], p[1]))
		It1_y_w = shift(It1_y, (p[0], p[1]))
	
		It1_w_rect = It1_w[XX, YY]	
		It_rect    = It[XX, YY]
		
		[It1_x_w_rect, It1_y_w_rect] = [It1_x_w[XX, YY], It1_y_w[XX, YY]]
		
		#cv2.imshow('template', 255*It_rect.astype('uint8'))
		#cv2.imshow('warp', 255*It1_w_rect.astype('uint8'))
		#cv2.imshow('xgrad', (255*It1_x_w_rect).astype('uint8'))
		#cv2.imshow('ygrad', (255*It1_y_w_rect).astype('uint8'))
		
		#print('xgrad: ', It1_x_w_rect)
		#if cv2.waitKey(5000) == ord('q'):break
		
		#np.gradient(It1_w_rect)
		
		b = (It1_w_rect - It_rect).flatten()
		
		A = np.matmul(np.stack((It1_x_w_rect.flatten(), It1_y_w_rect.flatten()), axis=1), warp_jacobian)
		
		H = np.matmul(A.T, A)
		
		del_p = np.linalg.lstsq(H, np.matmul(A.T, b), rcond=-1)[0]
		
		p = p + del_p
		
		norm_del_p = np.linalg.norm(del_p)
		
		#print('norm_p: ', p)
		
		if norm_del_p<threshold:
			break
	return -np.round(p)
