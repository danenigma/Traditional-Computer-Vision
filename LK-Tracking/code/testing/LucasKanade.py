import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import shift
#my imports
#import cv2

def LucasKanade(It, It1, rect, p0 = np.zeros(2)):
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
	
	#YY, XX = np.meshgrid(np.arange(rect[0], rect[2]), np.arange(rect[1],rect[3]))
	warp_jacobian = np.array([[1, 0],[0, 1]]).astype('float')

	[It1_x, It1_y] = np.gradient(It1) 
		
	H, W = It1.shape
	x = np.linspace(0, H, H)
	y = np.linspace(0, W, W)

	It_BiRect    = RectBivariateSpline(x, y, It)	
	It1_BiRect   = RectBivariateSpline(x, y, It1)	
	It1_x_BiRect = RectBivariateSpline(x, y, It1_x)
	It1_y_BiRect = RectBivariateSpline(x, y, It1_y)
	
	

	#line space
	xlin_t = np.linspace(rect[0], rect[2], rect[2]-rect[0])
	ylin_t = np.linspace(rect[1], rect[3], rect[3]-rect[1])
	xmesh_t, ymesh_t = np.meshgrid(xlin_t, ylin_t)
	It_rect = It_BiRect.ev(xmesh_t, ymesh_t)
		
	while True:	
			
		
		#It1_w   = shift(It1,   (p[1], p[0])) 
		#It1_x_w = shift(It1_x, (p[1], p[0]))
		#It1_y_w = shift(It1_y, (p[1], p[0]))
	
		#It1_w_rect = It1_w[XX, YY].reshape(-1)		
		#It_rect = It[XX, YY].reshape(-1)
		#It1_x_w_rect = It1_x_w[XX, YY].reshape(-1)
		#It1_y_w_rect = It1_y_w[XX, YY].reshape(-1)
		
		
		xlin = np.linspace(rect[0]+p[0], rect[2]+p[0], rect[2]-rect[0])
		ylin = np.linspace(rect[1]+p[1], rect[3]+p[1], rect[3]-rect[1])
		xmesh, ymesh = np.meshgrid(xlin, ylin)	
		print(xmesh.shape)	

		It1_w_rect   = It1_BiRect.ev(xmesh, ymesh)
		It1_x_w_rect = It1_x_BiRect.ev(xmesh, ymesh).flatten()
		It1_y_w_rect = It1_y_BiRect.ev(xmesh, ymesh).flatten()
		
		
		#A = np.matmul(np.stack((It1_x_w_rect, It1_y_w_rect), axis=1), warp_jacobian)
		
		#[dx, dy] = np.gradient(It_rect)
		#A = np.matmul(np.stack((dx.flatten(), dy.flatten()), axis=1), warp_jacobian)
		
		b = (It1_w_rect - It_rect).flatten()
		
		H = np.matmul(A.T, A)
		
		del_p = np.linalg.lstsq(H, np.matmul(A.T, b), rcond=-1)[0]
		p = p + del_p
		norm_del_p = np.dot(del_p, del_p)
		
		print('norm_p: ', del_p)
		
		if norm_del_p<threshold:
			break
	return p
