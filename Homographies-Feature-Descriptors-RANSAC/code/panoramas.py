import cv2
import os
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from planarH import ransacH, computeH
from BRIEF import briefLite,briefMatch,plotMatches

def getMask(im):
	
	mask = np.zeros((im.shape[0], im.shape[1]))
	mask[0,:] = 1
	mask[-1,:] = 1
	mask[:,0] = 1
	mask[:,-1] = 1
	mask = distance_transform_edt(1-mask)
	mask = mask/mask.max()

	return mask
def imageStitching(im1, im2, H2to1):
	'''
	Returns a panorama of im1 and im2 using the given 
	homography matrix

	INPUT
		Warps img2 into img1 reference frame using the provided warpH() function
		H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear
		         equation
	OUTPUT
		Blends img1 and warped img2 and outputs the panorama image
	'''
	#######################################
	# TO DO ...
	pano_im = cv2.warpPerspective(im2, H2to1, (im1.shape[1]+im2.shape[1], im1.shape[0]))
	pano_im[0:im1.shape[0], 0:im1.shape[1]] = im1 


	return pano_im


def imageStitching_noClip(im1, im2, H2to1):
	'''
	Returns a panorama of im1 and im2 using the given 
	homography matrix without cliping.
	''' 
	######################################
	# TO DO ...
	W  = 1700
	H2, W2 = im2.shape[0],im2.shape[1]
	H1, W1 = im1.shape[0],im1.shape[1]
	 
	im2_corners = np.array([[0, 0, 1],
							[W2, 0, 1],
							[0, H2, 1],
							[W2, H2, 1]]).T
	im1_corners = np.array([[0, 0, 1],
						[W1, 0, 1],
						[0, H1, 1],
						[W1, H1, 1]]).T
						
	im2_corners_proj  = np.matmul(H2to1, im2_corners)
	im2_corners_proj /= im2_corners_proj[2, :][None,:]
	
	H  = np.max(im2_corners_proj.astype('int')[1, :]) - np.min(im2_corners_proj.astype('int')[1, :])
	h4 = H - im2_corners_proj[1, 3]

	pano_im  = np.zeros((H, W, 3))
	M = np.array([[1., 0., 0.],
				   [0., 1., h4],
				   [0., 0., 1.]]).astype('float')
	
	warp_im1 = cv2.warpPerspective(im1, M, (W, H))
	im1_mask = getMask(im1)
	warp_im1_mask = cv2.warpPerspective(im1_mask, M, (W, H))
	
	warp_im2 = cv2.warpPerspective(im2, np.matmul(M, H2to1), (W, H))
	im2_mask = getMask(im2)
	warp_im2_mask = cv2.warpPerspective(im2_mask, np.matmul(M, H2to1), (W, H))
	
	
	#im2_mask = getMask(warp_im2)
	
	alpha = warp_im1_mask>warp_im2_mask
	
	
	alpha = np.repeat(alpha[:, :, np.newaxis], 3, axis=2).astype('float')
	#print(alpha.shape)
	#print(alpha)
	#pano_im =  alpha*warp_im2 + (1-alpha)*warp_im1
	#out = cv2.addWeighted(warp_im1_mask, 1, warp_im2_mask, 1, 0)
	pano_im = np.multiply(alpha, warp_im1).astype('uint8') + np.multiply(1-alpha, warp_im2).astype('uint8')
	#(_, warp_im2_mask) = cv2.threshold(cv2.cvtColor(warp_im2, cv2.COLOR_BGR2GRAY), 
	#												0, 255, cv2.THRESH_BINARY)

	#pano_im = cv2.add(pano_im, warp_im1, mask=np.bitwise_not(warp_im2_mask), 
	#									 dtype=cv2.CV_8U)

	#final_img = cv2.add(pano_im, warp_im2, dtype=cv2.CV_8U)
	return pano_im
	
def generatePanorama(im1, im2):

	locs1, desc1 = briefLite(im1)
	locs2, desc2 = briefLite(im2)
	matches = briefMatch(desc1, desc2)
	H2to1 = ransacH(matches, locs1, locs2, num_iter=5000, tol=2)		
	pano_im = imageStitching_noClip(im1, im2, H2to1)
	
	return pano_im

if __name__ == '__main__':

	im1 = cv2.imread('../data/incline_L.png')
	im2 = cv2.imread('../data/incline_R.png')

	pano_im = generatePanorama(im1, im2)

	cv2.imwrite('../results/panoImg.png', pano_im)
	cv2.imshow('panoramas', pano_im)
	cv2.waitKey(10000)
	cv2.destroyAllWindows()
	
