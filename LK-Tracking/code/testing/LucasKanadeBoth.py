import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import shift
import matplotlib.pyplot as plt
#my imports
import cv2

def LucasKanadeBoth(It, It1, rect, p0 = np.zeros(2)):
	# Input: 
	#	It: template image
	#	It1: Current image
	#	rect: Current position of the car
	#	(top left, bot right coordinates)
	#	p0: Initial movement vector [dp_x0, dp_y0]
	# Output:
	#	p: movement vector [dp_x, dp_y]
	# Put your implementation here
	
	threshold = 0.01
	p = p0
	
	It = np.float32(It)/np.max(It)
	It1 = np.float32(It1)/np.max(It1)
	
	
	#YY, XX = np.meshgrid(np.linspace(rect[0], rect[2]+1, rect[2]-rect[0]+1), 
	#					 np.linspace(rect[1], rect[3]+1, rect[3]-rect[1]+1))
	
	YY, XX = np.meshgrid(np.arange(rect[0], rect[2]+1), np.arange(rect[1], rect[3]+1))


	print('XX:', XX)
	[It1_x, It1_y] = np.gradient(It1) 
	H, W = It.shape
	
	x = np.arange(0, H)
	y = np.arange(0, W)
	print(len(x), It.shape)
	
	It0_BiRect = RectBivariateSpline(x, y, It)	
		

	while True:	
			
		
		It1_w   = shift(It1,   (p[1], p[0])) 
		It1_x_w = shift(It1_x, (p[1], p[0]))
		It1_y_w = shift(It1_y, (p[1], p[0]))
	
		It1_BiRect   = RectBivariateSpline(x, y, It1_w)	
		It1_x_BiRect = RectBivariateSpline(x, y, It1_x_w)	
		It1_y_BiRect = RectBivariateSpline(x, y, It1_y_w)	
		
		It1_w_rect   = It1_BiRect.ev(XX, YY)	
		It1_x_w_rect = It1_x_BiRect.ev(XX, YY)	
		It1_y_w_rect = It1_y_BiRect.ev(XX, YY)	
		It_rect    = It0_BiRect.ev(XX, YY)
		
		
		cv2.imshow('template', It_rect)
		cv2.imshow('warp', It1_w_rect)

		cv2.waitKey(1000)
		
		b = It_rect.flatten() - It1_w_rect.flatten()  
		
		print('b: ', np.sum(b))
		A = np.stack((It1_x_w_rect.flatten(), It1_y_w_rect.flatten()), axis=1)
		
		H = np.matmul(A.T, A)
		
		del_p = np.linalg.inv(H) @ (A.T @ b)
		print(del_p)
		p = p + del_p
		
		norm_del_p = np.linalg.norm(del_p)
		
		print('norm_p: ', norm_del_p)
		
		if norm_del_p<threshold:
			break
	print('p: ', p)
	return p
