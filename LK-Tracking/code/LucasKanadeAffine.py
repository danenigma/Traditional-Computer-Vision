import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import affine_transform
import cv2

def LucasKanadeAffine(It, It1):
	# Input: 
	#	It: template image
	#	It1: Current image
	# Output:
	#	M: the Affine warp matrix [2x3 numpy array]
	# put your implementation here
	M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

	threshold = 0.5

	p = np.array([0.,0.,0.,0.,0.,0.]).T

	It = np.float32(It)/np.max(It)
	It1 = np.float32(It1)/np.max(It1)

	H, W = It.shape

	[It1_x, It1_y] = np.gradient(It1) 
	

	while True:	
		

		It1_w   = affine_transform(It1,  M[:2,:2], offset =[M[0, 2], M[1, 2]], cval=-1)
		It1_x_w = affine_transform(It1_x,M[:2,:2], offset =[M[0, 2], M[1, 2]], cval=-1)
		It1_y_w = affine_transform(It1_y,M[:2,:2], offset =[M[0, 2], M[1, 2]], cval=-1)

		x, y  = np.where(It1_w!=-1)
		 
		It1_w_rect  = It1_w[x, y]	
		It_rect     = It[x, y]
	
		[It1_x_w_rect, It1_y_w_rect] = [It1_x_w[x, y], It1_y_w[x, y]]
	

	
		b = It1_w_rect - It_rect
	
		A = np.stack((It1_x_w_rect*x,
					  It1_x_w_rect*y,
					  It1_x_w_rect, 
					  It1_y_w_rect*x,
					  It1_y_w_rect*y,
					  It1_y_w_rect), axis=1)
	
		H = np.matmul(A.T, A)
	
		del_p = np.linalg.lstsq(H, np.matmul(A.T, b), rcond=-1)[0]
	
		p = p + del_p
		M[0, 0] = p[0]+1
		M[0, 1] = p[1]
		M[0, 2] = p[2]
		M[1, 0] = p[3]
		M[1, 1] = p[4]+1
		M[1, 2] = p[5]
		norm_del_p = np.linalg.norm(del_p)
	
		print('norm_p: ', norm_del_p)
		#print('M: ', M)
		
		if norm_del_p<threshold:
			print('breaking .. ')
			break

	return M
