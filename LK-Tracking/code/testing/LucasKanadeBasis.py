import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import shift

def LucasKanadeBasisNew(It, It1, rect, bases):
	# Input: 
	#	It: template image
	#	It1: Current image
	#	rect: Current position of the car
	#	(top left, bot right coordinates)
	#	p0: Initial movement vector [dp_x0, dp_y0]
	# Output:
	#	p: movement vector [dp_x, dp_y]
	# Put your implementation here
	
	threshold = 0.001
	p = np.zeros(2)
	
	It = np.float32(It)/np.max(It)
	It1 = np.float32(It1)/np.max(It1)
	
	
	YY, XX = np.meshgrid(np.arange(rect[0], rect[2]+1), np.arange(rect[1],rect[3]+1))
	warp_jacobian = np.array([[1, 0],[0, 1]]).astype('float')

	[It1_x, It1_y] = np.gradient(It1) 
		
	B   = bases.reshape(-1, bases.shape[-1])

	I = np.eye(B.shape[0])
	
	I_BBT = I - B @ B.T

	while True:	
			
		
		It1_w   = shift(It1,   (p[0], p[1])) 
		It1_x_w = shift(It1_x, (p[0], p[1]))
		It1_y_w = shift(It1_y, (p[0], p[1]))
	
		It1_w_rect = It1_w[XX, YY]	
		It_rect    = It[XX, YY]
		
		[It1_x_w_rect, It1_y_w_rect] = [It1_x_w[XX, YY], It1_y_w[XX, YY]]
		

		
		b = (It1_w_rect - It_rect).flatten()
		
		A = np.stack((It1_x_w_rect.flatten(), It1_y_w_rect.flatten()), axis=1)
		
		A_star = I_BBT @ A
		
		b_star = I_BBT @ b 
	
		H = A_star.T @ A_star

		del_p = np.linalg.lstsq(H, A_star.T @ b_star, rcond=-1)[0]
		
		p = p + del_p
		
		norm_del_p = np.linalg.norm(del_p)
		
		print('norm_p: ', norm_del_p)
		
		if norm_del_p<threshold:
			break
	return -np.round(p)
