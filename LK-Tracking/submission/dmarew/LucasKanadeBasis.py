import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import shift
import matplotlib.pyplot as plt
#my imports
import cv2

def LucasKanadeBasis(It, It1, rect, bases):
	# Input: 
	#	It: template image
	#	It1: Current image
	#	rect: Current position of the car
	#	(top left, bot right coordinates)
	#	p0: Initial movement vector [dp_x0, dp_y0]
	# Output:
	#	p: movement vector [dp_x, dp_y]
	# Put your implementation here

	threshold = 0.0001
	p = np.zeros(2).astype('float')
	
	It  = np.float32(It)/np.max(It)
	It1 = np.float32(It1)/np.max(It1)

	[It1_x, It1_y] = np.gradient(It1) 
	rect = rect.astype('float')
	
	H, W = It1.shape
	
	x = np.arange(0, H)
	y = np.arange(0, W)

	It0_BiRect = RectBivariateSpline(x, y, It)	
	It1_BiRect = RectBivariateSpline(x, y, It1)	
	It1_x_BiRect = RectBivariateSpline(x, y, It1_x)	
	It1_y_BiRect = RectBivariateSpline(x, y, It1_y)	

	[H_b, W_b, T] = bases.shape

	y_range = np.linspace(rect[0], rect[2] + 1, W_b)
	x_range = np.linspace(rect[1], rect[3] + 1, H_b)		
	[Y, X] = np.meshgrid(y_range + p[1], x_range + p[0])
						 	
	template = It0_BiRect.ev(X, Y)

	B = bases.reshape(-1, bases.shape[-1])

	I = np.eye(B.shape[0])
	
	I_BBT = I - B @ B.T

	
	while True:	

		y_range = np.linspace(rect[0], rect[2] + 1, W_b)
		x_range = np.linspace(rect[1], rect[3] + 1, H_b)		
		[Y, X] = np.meshgrid(y_range + p[1], x_range + p[0])
	
	 	
		It1_w_rect   = It1_BiRect.ev(X, Y)
		It1_x_w_rect = It1_x_BiRect.ev(X, Y)
		It1_y_w_rect = It1_y_BiRect.ev(X, Y)

		b = (template - It1_w_rect).flatten()
		A = np.stack((It1_x_w_rect.flatten(), It1_y_w_rect.flatten()), axis=1)

		A_star = I_BBT @ A

		b_star = I_BBT @ b 

		H = np.matmul(A_star.T, A_star)

		del_p = np.linalg.inv(H) @ (A_star.T @ b_star)
		p = p + del_p
		norm_del_p = np.linalg.norm(del_p)
		if norm_del_p<threshold:
			break

	return p
