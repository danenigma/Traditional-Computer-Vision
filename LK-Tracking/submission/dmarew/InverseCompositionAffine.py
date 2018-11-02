import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import affine_transform

def InverseCompositionAffine(It, It1):
	# Input: 
	#	It: template image
	#	It1: Current image

	# Output:
	#	M: the Affine warp matrix [2x3 numpy array]

	# put your implementation here
	M = np.array([[1.0, 0.0, 0.0],
				  [0.0, 1.0, 0.0], 
				  [0.0, 0.0, 1.]]).astype('float')

	threshold = 0.1



	It = np.float32(It)/np.max(It)
	It1 = np.float32(It1)/np.max(It1)

	H, W = It.shape

	[It_x, It_y] = np.gradient(It) 
	[x, y] = np.meshgrid(np.arange(W), np.arange(H))
	[x, y] = x.flatten(), y.flatten()
	
	A = np.stack((It_x.flatten()*x,
				  It_x.flatten()*y,
				  It_x.flatten(), 
				  It_y.flatten()*x,
				  It_y.flatten()*y,
				  It_y.flatten()), axis=1)
				  
	H_inv = np.linalg.inv(A.T @ A)
	M_del = M.copy()			  
	
	while True:	
	

		It1_w   = affine_transform(It1,  M[:2,:2], offset =[M[0, 2], M[1, 2]])
		
		It1_w_flat = It1_w.flatten()
		It_flat = It.flatten()

		valid_indices = np.where(It1_w_flat!=0)[0]
					 		
		b = (It1_w_flat[valid_indices] - It_flat[valid_indices])
		
		A_valid = A[valid_indices, :]

		del_p = H_inv @ (A_valid.T @ b)

		M_del[0, 0] = del_p[0]+1
		M_del[0, 1] = del_p[1]
		M_del[0, 2] = del_p[2]
		M_del[1, 0] = del_p[3]
		M_del[1, 1] = del_p[4]+1
		M_del[1, 2] = del_p[5]
		
		M = M @ np.linalg.inv(M_del)
	
		norm_del_p = np.linalg.norm(del_p)

		print('norm_p: ', norm_del_p)
		#print('M: ', M)
	
		if norm_del_p<threshold:
			print('breaking .. ')
			break

	return M[:2, :]
