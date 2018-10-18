import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import shift

def LucasKanadeBasis(It, It1, rect, bases):
	# Input: 
	#	It: template image
	#	It1: Current image
	#	rect: Current position of the car
	#	(top left, bot right coordinates)
	#	bases: [n, m, k] where nxm is the size of the template.
	# Output:
	#	p: movement vector [dp_x, dp_y]

    # Put your implementation here
    	
	threshold = 0.00001
	p = np.zeros(2)
	
	It = np.float32(It)/np.max(It)
	It1 = np.float32(It1)/np.max(It1)
	
	
	YY, XX = np.meshgrid(np.arange(rect[0], rect[2]+1), np.arange(rect[1],rect[3]+1))
	warp_jacobian = np.array([[1, 0],[0, 1]]).astype('float')

	[It1_x, It1_y] = np.gradient(It1) 
		
	
	while True:	
			
		
		It1_w   = shift(It1,   (p[0], p[1])) 
		It1_x_w = shift(It1_x, (p[0], p[1]))
		It1_y_w = shift(It1_y, (p[0], p[1]))
	
		It1_w_rect = It1_w[XX, YY]	
		It_rect    = It[XX, YY]
		
		[It1_x_w_rect, It1_y_w_rect] = [It1_x_w[XX, YY], It1_y_w[XX, YY]]
		
		bases_mat = bases.reshape(-1, bases.shape[-1])
		
		A = np.stack([It1_x_w_rect.flatten(), It1_y_w_rect.flatten()], axis=1)
		#print('del_I',del_I.shape, It_rect.shape)
		
		b = (It1_w_rect - It_rect).flatten()
		
		W = np.matmul(bases_mat.T, A)
		print(np.linalg.norm(np.matmul(bases_mat, W)))
		A = A -  np.matmul(bases_mat, W)
		
		H = np.matmul(A.T, A)
		
		del_p = np.linalg.lstsq(H, np.matmul(A.T, b), rcond=-1)[0]
		
		p = p + del_p
		
		norm_del_p = np.linalg.norm(del_p)
		
		#print('norm_p: ', p)
		
		if norm_del_p<threshold:
			break
	return -np.round(p)
    
