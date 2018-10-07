import numpy as np
import cv2
from BRIEF import briefLite, briefMatch
from scipy.spatial.distance import cdist


def computeH(p1, p2):
	'''
	INPUTS:
		p1 and p2 - Each are size (2 x N) matrices of corresponding (x, y)'  
		         coordinates between two images
	OUTPUTS:
	 H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear 
		    equation
	'''
	assert(p1.shape[1]==p2.shape[1])
	assert(p1.shape[0]==2)
	#############################
	# TO DO ...
	N = p1.shape[1]
	A = np.zeros((2*N, 9)).astype('int')
	#constructing A matrix
	for i in range(N):
		[x, y] = p1[:, i]
		[u, v] = p2[:, i]
		A[2*i, :] = np.array([0, 0, 0,-u ,-v,-1,y*u,y*v, y])
		A[2*i+1, :] = np.array([u, v, 1, 0, 0, 0, -x*u, -x*v, -x])
		
	U, S, V = np.linalg.svd(A, True)
	
	H2to1 = np.reshape(V[-1,:],(3,3))

	return H2to1

def ransacH(matches, locs1, locs2, num_iter=5000, tol=2):
	'''
	Returns the best homography by computing the best set of matches using
	RANSAC
	INPUTS
		locs1 and locs2 - matrices specifying point locations in each of the images
		matches - matrix specifying matches between these two sets of point locations
		nIter - number of iterations to run RANSAC
		tol - tolerance value for considering a point to be an inlier

	OUTPUTS
		bestH - homography matrix with the most inliers found during RANSAC
	''' 
	###########################
	# TO DO ...
	p = matches.shape[0]
	print('P: ', p, matches.shape)
	bestH = None
	bestH_inlier_count = 0
	bestH_inliers = None
	
	matchedLoc2 = locs2[matches[:, 1]] 
	matchedLoc1 = locs1[matches[:, 0]][:, :2] 

	
	matchedLoc2Homo = np.copy(matchedLoc2.T)
	matchedLoc2Homo[2, :] = np.ones((1, matchedLoc2Homo.shape[1]))
	
	for iter in range(num_iter):
		randPtsIdx = np.random.choice(p,4,replace=False)
		
		locIdx = matches[randPtsIdx]
		randLoc1, randLoc2 = locs1[locIdx[:, 0]][:, :2], locs2[locIdx[:, 1]][:, :2]
		H = computeH(randLoc1.T, randLoc2.T)
		
		loc2Proj = np.matmul(H, matchedLoc2Homo)
		#normalize by lamda
		loc2ProjNorm = (loc2Proj / loc2Proj[2, :][None,:])[:2, :]

		dst = np.diag(cdist(loc2ProjNorm.T, matchedLoc1))
		inliers = dst<tol
		inliers_count = np.sum(dst<tol)

		if inliers_count >  bestH_inlier_count:
			print('new max: ', inliers_count)
			bestH_inlier_count = inliers_count
			bestH = H
			bestH_inliers = inliers
			
	bestH = computeH(matchedLoc1[bestH_inliers].T, matchedLoc2[bestH_inliers][:, :2].T)
		
	return bestH


if __name__ == '__main__':
	N = 4
	p1 = np.random.randint(0, 130, (2, N))
	p2 = np.random.randint(0, 130, (2, N))
	#H2to1 = computeH(p1, p2)

	im1 = cv2.imread('../data/model_chickenbroth.jpg')
	im2 = cv2.imread('../data/chickenbroth_01.jpg')
	locs1, desc1 = briefLite(im1)
	locs2, desc2 = briefLite(im2)

	matches = briefMatch(desc1, desc2)
	ransacH(matches, locs1, locs2, num_iter=5000, tol=2)

