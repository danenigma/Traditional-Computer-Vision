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
	mask = mask/mask.max(0)

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
	h4 = 200 
	W  = 1700
	H2, W2 = im2.shape[0],im2.shape[1] 
	
	[W2_, H2_, lamda] = np.matmul(H2to1, np.array([W2, H2, 1]).T)
	[W20_, H20_, lamda0] = np.matmul(H2to1, np.array([0, 0, 1]).T)
	
	W2_/=lamda
	H2_/=lamda
	W20_/=lamda0
	H20_/=lamda0
	
	print(W20_, H20_, W2_, H2_)
	h4 = -H20_ + 100
	H = int(H2 + h4)
	pano_im  = np.zeros((H, W, 3))
	M = np.array([[1, 0., 0.],
				   [0., 1, h4],
				   [0., 0., 1.]]).astype('float')
	warp_im1 = cv2.warpPerspective(im1, M, (W, H))
	im1_mask = getMask(im1)
	warp_im1_mask = cv2.warpPerspective(im1_mask, M, (W, H))
	
	warp_im2 = cv2.warpPerspective(im2, np.matmul(M, H2to1), (W, H))
	im2_mask = getMask(im2)
	warp_im2_mask = cv2.warpPerspective(im2_mask, np.matmul(M, H2to1), (W, H))
	
	
	#im2_mask = getMask(warp_im2)
	alpha = warp_im2_mask>warp_im2_mask
	print(alpha)
	#alpha = np.repeat(alpha[:, :, np.newaxis], 3, axis=2)
	#print(alpha.shape)

	#pano_im =  alpha*warp_im2 + (1-alpha)*warp_im1
	out = cv2.addWeighted(warp_im1, 0.5, warp_im2, 0.5,0)
	"""
	(ret,data_map) = cv2.threshold(cv2.cvtColor(warp_im2, cv2.COLOR_BGR2GRAY), 
	0, 255, cv2.THRESH_BINARY)
	pano_im = cv2.add(pano_im, warp_im1, mask=np.bitwise_not(data_map), dtype=cv2.CV_8U)
	# Now add the warped image
	final_img = cv2.add(pano_im, warp_im2, dtype=cv2.CV_8U)
	"""
	return out#warp_im2_mask# final_img


if __name__ == '__main__':
	im1 = cv2.imread('../data/incline_L.png')
	im2 = cv2.imread('../data/incline_R.png')

	#plotMatches(im1,im2,matches,locs1,locs2)


	#H2to1 = computeH(locs1, locs2)
	H_file = 'H2to1.npy'
	H2to1  = None
	if os.path.isfile(H_file):
		H2to1= np.load(H_file)
	else:
		print('RANSACing....')
		
		locs1, desc1 = briefLite(im1)
		locs2, desc2 = briefLite(im2)
		matches = briefMatch(desc1, desc2)
		H2to1 = ransacH(matches, locs1, locs2, num_iter=5000, tol=2)
		np.save('H2to1.npy', H2to1)

	#pano_im = imageStitching_noClip(im1, im2, H2to1)
	pano_im = imageStitching_noClip(im1, im2, H2to1)

	#print(H2to1)
	#-----TODO---- DONT FORGET TO CHANGE 
	#cv2.imwrite('../results/panoImgTest.png', pano_im)
	cv2.imshow('panoramas', pano_im)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
