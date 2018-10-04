import cv2
import os
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from planarH import ransacH, computeH
from BRIEF import briefLite,briefMatch,plotMatches

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
    return pano_im


if __name__ == '__main__':
	im1 = cv2.imread('../data/incline_L.png')
	im2 = cv2.imread('../data/incline_R.png')

	locs1, desc1 = briefLite(im1)
	locs2, desc2 = briefLite(im2)
	matches = briefMatch(desc1, desc2)
	#plotMatches(im1,im2,matches,locs1,locs2)


	#H2to1 = computeH(locs1, locs2)
	H_file = 'H2to1.npy'
	H2to1  = None
	if os.path.isfile(H_file):
		H2to1= np.load(H_file)
	else:
		print('RANSACing....')
		H2to1 = ransacH(matches, locs1, locs2, num_iter=5000, tol=2)
		np.save('H2to1.npy', H2to1)

	#pano_im = imageStitching_noClip(im1, im2, H2to1)
	pano_im = imageStitching(im1, im2, H2to1)

	#print(H2to1)
	#-----TODO---- DONT FORGET TO CHANGE 
	#cv2.imwrite('../results/panoImgTest.png', pano_im)
	cv2.imshow('panoramas', pano_im)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
