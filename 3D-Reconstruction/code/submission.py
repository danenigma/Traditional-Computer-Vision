"""
Homework4.
Replace 'pass' by your implementation.
"""
import numpy as np
import helper
# Insert your package here
from sympy import *

'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix
'''
def eightpoint(pts1, pts2, M):
	# Replace pass by your implementation
	T = np.eye(3) / M
	T[2, 2] = 1;
	pts1 = pts1.astype('float')/M
	pts2 = pts2.astype('float')/M 
	 
	A = np.vstack([
		 pts1[:, 0]*pts2[:, 0],pts1[:, 0]*pts2[:, 1], pts1[:, 0],
		 pts1[:, 1]*pts2[:, 0],pts1[:, 1]*pts2[:, 1], pts1[:, 1],
		 pts2[:, 0],pts2[:, 1], np.ones(pts1.shape[0])		 
		]).T

	[U, S, V] = np.linalg.svd(A)		
	F = np.reshape(V[-1,:], (3,3))
	
	F = helper.refineF(F, pts1, pts2)	
	F = T.T @ F @ T

	
	return F



'''
Q2.2: Seven Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: Farray, a list of estimated fundamental matrix.
'''
def sevenpoint(pts1, pts2, M):
    # Replace pass by your implementation
	T = np.eye(3) / M
	T[2, 2] = 1;
	pts1 = pts1.astype('float')/M
	pts2 = pts2.astype('float')/M 
	
	Fs = [] 
	A = np.vstack([
		 pts1[:, 0]*pts2[:, 0],pts1[:, 0]*pts2[:, 1], pts1[:, 0],
		 pts1[:, 1]*pts2[:, 0],pts1[:, 1]*pts2[:, 1], pts1[:, 1],
		 pts2[:, 0],pts2[:, 1], np.ones(pts1.shape[0])		 
		]).T

	[U, S, V] = np.linalg.svd(A)			

	F1 = np.reshape(V[-1,:], (3,3))
	F2 = np.reshape(V[-2,:], (3,3))
	
	alpha = Symbol('alpha')
	
	eqn   = Matrix(F1 + alpha*F2).det()

	solns = roots(eqn)
	
	for i, sol in enumerate(solns):
		
		if re(sol)==sol:
			sol = float(sol)
			F = F1 + sol*F2
			F = helper.refineF(F, pts1, pts2)		
			Fs.append(T.T @ F @ T)

	return Fs

'''
Q3.1: Compute the essential matrix E.
    Input:  F, fundamental matrix
            K1, internal camera calibration matrix of camera 1
            K2, internal camera calibration matrix of camera 2
    Output: E, the essential matrix
'''
def essentialMatrix(F, K1, K2):
    # Replace pass by your implementation
    return K2.T @ F @ K1 
    #pass


'''
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.
'''
def triangulate(C1, pts1, C2, pts2):
    # Replace pass by your implementation
    
	P = []

	error = 0
	
	for pt1, pt2 in zip(pts1, pts2):
		x1, y1, x2, y2 = pt1[0], pt1[1], pt2[0], pt2[1] 
	
		A = np.vstack([(x1*C1[2, :]-C1[0, :]),
			 (y1*C1[2, :]-C1[1, :]),
			 (x2*C2[2, :]-C2[0, :]),
			 (y2*C2[2, :]-C2[1, :])])

		[U, S, V] = np.linalg.svd(A)
		w = V[-1,:]/V[-1,-1]
		
		p1_reproj = C1 @ w
		p2_reproj = C2 @ w
		
		p1_reproj = p1_reproj/p1_reproj[-1]
		p2_reproj = p2_reproj/p2_reproj[-1]
		error += np.linalg.norm(p1_reproj[:2]- pt1) + np.linalg.norm(p2_reproj[:2]- pt2)		
		P.append(w[:3])
		
	P = np.vstack(P)
	
	return P, error
'''
Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2

'''
def epipolarCorrespondence(im1, im2, F, x1, y1):
    # Replace pass by your implementation
    
    pass

'''
Q5.1: RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
    Output: F, the fundamental matrix
'''
def ransacF(pts1, pts2, M):
    # Replace pass by your implementation

	p = pts1.shape[0]

	bestF = None
	bestF_inlier_count = 0
	bestF_inliers = None
	
	for iter in range(num_iter):
		randPtsIdx = np.random.choice(p, 7, replace=False)
		

		randLoc1, randLoc2 = pts1[randPtsIdx, :], pts2[randPtsIdx, :] 
		F = sevenpoint(randLoc1, randLoc2, M)

		# get essential matrix
		# project the to 
		# compute distace 
		# if distance 
		dst = np.diag(cdist(loc2ProjNorm.T, matchedLoc1))
		inliers = dst<tol
		inliers_count = np.sum(dst<tol)

		if inliers_count >  bestF_inlier_count:
			#print('new max: ', inliers_count)
			bestF_inlier_count = inliers_count
			bestF = F
			bestF_inliers = inliers
			
	bestF = sevenpoint(pts1[bestF_inliers, :], pts2[bestF_inliers, :], M)
		
	return bestF


'''
Q5.2: Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
'''
def rodrigues(r):
    # Replace pass by your implementation
    pass

'''
Q5.2: Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
'''
def invRodrigues(R):
    # Replace pass by your implementation
    pass

'''
Q5.3: Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2.
    Output: residuals, the difference between original and estimated projections
'''
def rodriguesResidual(K1, M1, p1, K2, p2, x):
    # Replace pass by your implementation
    pass

'''
Q5.3 Bundle adjustment.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 1
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 1
            P2, the optimized 3D coordinates of points
'''
def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    # Replace pass by your implementation
    pass
    
if __name__ == '__main__':
	data = np.load('../data/some_corresp.npz')
	pts1, pts2 = data['pts1'], data['pts2']
	M = 400
	eightpoint(pts1, pts2, M)
