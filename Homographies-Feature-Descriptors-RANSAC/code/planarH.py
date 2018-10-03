import numpy as np
import cv2
from BRIEF import briefLite, briefMatch

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
	
	for i in range(N):
		[x, y] = p1[:, i]
		[u, v] = p2[:, i]
		
		A[2*i, :] = np.array([0, 0, 0,-u ,-v,-1,y*u,y*v, y])
		A[2*i+1, :] = np.array([u, v, 1, 0, 0, 0, -x*u, -x*v, -x])
		
	U, S, V = np.linalg.svd(A, False)
	#print(np.sum(np.dot(A, V[-1,:])))
	h33 = V[-1,-1]
	H2to1 = np.reshape(V[-1,:],(3,3))
	print(H2to1)
	print(np.matmul(H2to1.T, H2to1))
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
    for iter in range(num_iter):
    	pass
    	
    return bestH
        
    

if __name__ == '__main__':
	N = 4
	p1 = np.random.randint(0, 130, (2, N))
	p2 = np.random.randint(0, 130, (2, N))
	H2to1 = computeH(p1, p2)

	#im1 = cv2.imread('../data/model_chickenbroth.jpg')
	#im2 = cv2.imread('../data/chickenbroth_01.jpg')
	#locs1, desc1 = briefLite(im1)
	#locs2, desc2 = briefLite(im2)

	#matches = briefMatch(desc1, desc2)
	#ransacH(matches, locs1, locs2, num_iter=5000, tol=2)

