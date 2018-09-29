import imageio
import numpy as np
from numpy.linalg import inv
def warp(im, A, output_shape):
	""" Warps (h,w) image im using affine (3,3) matrix A
	producing (output_shape[0], output_shape[1]) output image
	with warped = A*input, where warped spans 1...output_size.
	Uses nearest neighbor interpolation."""

	output = np.zeros((output_shape[0], output_shape[1]))
	img_w, img_h = im.shape 
	
	for i in range(output_shape[0]):
		for j in range(output_shape[1]):
			i_p, j_p = np.dot(inv(A), np.array([i, j, 1])).tolist()[:2]
			i_p, j_p = int(i_p), int(j_p)
			if (i_p < img_w and i_p > 0) and (j_p > 0 and j_p < img_h):
				
				output[i][j] = im[i_p][j_p]
	x = np.linspace(0, output_shape[0]-1, output_shape[0])
	y = np.linspace(0, output_shape[1]-1, output_shape[1])
	xx, yy = np.meshgrid(x,y)
	A_inv = inv(A)

	xx_p = A_inv[0][0]*xx + A_inv[0][1]*yy + A_inv[0][2]
	yy_p = A_inv[1][0]*xx + A_inv[1][1]*yy + A_inv[1][2]
	 
	#print(xx_p)	
		
	return output

