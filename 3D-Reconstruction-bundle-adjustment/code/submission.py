"""
Homework4.
Replace 'pass' by your implementation.
"""
import numpy as np
import helper
# Insert your package here
from sympy import *
from scipy.ndimage.filters import gaussian_filter
from scipy.optimize import leastsq, minimize
import cv2

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
	#fun = lambda a: np.linalg.det(a * F1 + (1 - a) * F2)
	#a0=fun(0)
	#a1=(fun(1)−fun(−1))/3−(fun(2)−fun(−2))/12
	#a2=0.5fun(1)+0.5fun(−1)−fun(0)
	#a0+a1x+a2x2+a3x3=fun(x)
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
		#print(w[2]) #p1_reproj[:2], pt1, p1_reproj[:2]- pt1) #np.linalg.norm(p1_reproj[:2]- pt1)**2)
		error += (np.linalg.norm(p1_reproj[:2]- pt1)**2 + np.linalg.norm(p2_reproj[:2]- pt2)**2)		
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

	sy, sx, _ = im2.shape
	v = np.array([x1, y1, 1])
	l = F.dot(v)
	s = np.sqrt(l[0]**2+l[1]**2)

	if s==0:
		error('Zero line vector in displayEpipolar')
	l = l/s

	if l[0] != 0:
		ye = sy-1;
		ys = 0

		Y = np.arange(np.ceil(min(ys,ye)), np.ceil(max(ys,ye)))		
		X = np.round(-(l[1]*Y + l[2])/l[0])
		
				
	else:
		xe = sx-1
		xs = 0
		
		X = np.arange(np.ceil(min(xs,xe)), np.ceil(max(xs,xe)))		
		Y = np.round(-(l[0]*X + l[2])/l[1])

	X = np.round(X).astype('int')
	Y = np.round(Y).astype('int')

	
	S = 15
	patch1 = getPatch(im1, y1, x1, S)
	x2, y2 = None, None	 
	min_dist = np.inf
	g_kernel = gaussian(S, 15)
	g_kernel_3d = np.repeat(g_kernel[:, :, np.newaxis], 3, axis=2)

	for i, j in zip(X, Y):
		patch2 = getPatch(im2, j, i, S)		
		if patch2 is None:continue
		dist = np.linalg.norm(g_kernel_3d*(patch1-patch2))
		if dist <  min_dist :
			min_dist = dist 
			[x2, y2] = i, j



			
	return x2, y2

def getPatch(im, x, y, S):
	h, w = im.shape[:2]
	
	if ((x + S//2+1)<h and (x - S//2)>0 and (y + S//2+1)<w and (y - S//2)>0 ):
		return im[x-S//2:x+S//2+1, y-S//2:y+S//2+1, :]
	else:#if not valid patch
		return None	
def gaussian(size, sigma):
	X = np.linspace(-size//2 + 1, size//2 + 1, size)
	Y = np.linspace(-size//2 + 1, size//2 + 1, size)
	XX, YY = np.meshgrid(X, Y)
	g = np.exp(-((XX**2 + YY**2)/(2.0*sigma**2)))
	return g/np.sum(g)	

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
	tol = 0.001
	num_iter = 5000
	pts1_h = np.hstack((pts1, np.ones((p, 1))))
	pts2_h = np.hstack((pts2, np.ones((p, 1))))
	
	for iter in range(num_iter):
		randPtsIdx = np.random.choice(p, 7, replace=False)
		randLoc1, randLoc2 = pts1_h[randPtsIdx, :], pts2_h[randPtsIdx, :] 
		Fs = sevenpoint(randLoc1[:, :2], randLoc2[:, :2], M)
		
		for F in Fs:
			dst = np.diag(pts2_h @ F @ pts1_h.T)
			inliers =  np.abs(dst) < tol
			inliers_count = np.sum(inliers)
		
			if inliers_count >  bestF_inlier_count:
				bestF_inlier_count = inliers_count
				bestF_inliers = inliers
				bestF = F
		print('#'*100)	
		print('iter: ', iter, ' inliers: ', int(100*inliers_count/p),'/', int(100*bestF_inlier_count/p))	
		print('#'*100)
		if (bestF_inlier_count/p) >=.75:
			break

	bestFs = sevenpoint(pts1[bestF_inliers, :2], pts2[bestF_inliers, :2], M)
	
	for F in bestFs:
		dst = np.diag(pts2_h @ F @ pts1_h.T)
		inliers =  np.abs(dst) < tol
		inliers_count = np.sum(inliers)
		bestF_inlier_count = 0
	
		if inliers_count >  bestF_inlier_count:
			#print('new max: ', inliers_count)
			bestF_inlier_count = inliers_count
			bestF = F
			bestF_inliers = inliers
	
	bestF = helper.refineF(bestF, pts1[bestF_inliers, :2], pts2[bestF_inliers, :2])		
		
	
	return bestF, np.where(bestF_inliers)


'''
Q5.2: Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
'''
def rodrigues(r):
	# Replace pass by your implementation
	r = r.reshape(3, 1)	
	#return cv2.Rodrigues(r.astype('float'))[0]
	th  = np.linalg.norm(r)


	if th == 0:
		return np.eye(3)

	u = r/th
	
	a1, a2, a3 = u[0], u[1], u[2]
	 
	u_x = np.array([[0,  -a3, a2],
				   [a3,  0, -a1],
				   [-a2, a1, 0]])
				   
	#R = np.eye(3) + np.sin(th)*rx + (1 - np.cos(th) )*(r_n @ r_n.T - np.eye(3))
	R = np.eye(3)*np.cos(th) + (1 - np.cos(th))*(u @ u.T) + np.sin(th)*u_x 
	
	return R

'''
Q5.2: Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
'''
def invRodrigues(R):
	#https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation

	th = np.arccos((np.trace(R) - 1)/2)
	#return cv2.Rodrigues(R.astype('float'))[0].reshape(-1)
	if th == 0 or np.isnan(th):
		return np.zeros(3).reshape(-1, 1), cv2.Rodrigues(R.astype('float'))[0]
	else:
		'''
		r = (1/(2*np.sin(th)))*np.array([R[2, 1] - R[1, 2],
			 							 R[0, 2] - R[2, 0],
		 								 R[1, 0] - R[0, 1]]).T			 		
		'''
		A = (R - R.T)/2
		a32, a13, a21 = A[2,1], A[0,2], A[1,0]
		
		rho = np.array([a32, a13, a21]).T
		s  = np.linalg.norm(rho)
		c  = (np.trace(R) - 1)/2
		u  = rho/s
		th = np.arctan2(s, c)
		r  = u*th
		
				 
	return r
	   

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
#########ASK Question on Piazza
# used to be rodriguesResidual(K1, M1, p1, K2, p2, x)
def rodriguesResidual(K1, M1, p1, K2, p2, x):
	N = p1.shape[0]

	P  = x[:-6].reshape(N, 3)
	r2 = x[-6:-3].reshape(3, 1) 
	t2 = x[-3:].reshape(3, 1)
	R2 = rodrigues(r2)

	M2 = np.concatenate([R2, t2], axis=1)

	P_h = np.hstack((P, np.ones((N,1))))


	p1_hat    = (K1 @ M1) @ P_h.T
	p2_hat    = (K2 @ M2) @ P_h.T
	p1_hat    = (p1_hat/p1_hat[2, :])[:2, :].T
	p2_hat    = (p2_hat/p2_hat[2, :])[:2, :].T

	residuals = np.concatenate([(p1-p1_hat).reshape([-1]),
								(p2-p2_hat).reshape([-1])])

	return residuals

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
	N = p1.shape[0]

	R2 = M2_init[:, :3]
	t2 = M2_init[:, 3]
	r2 = invRodrigues(R2)
	
	x_init = np.concatenate([P_init.reshape([-1]), r2, t2])
	rod_func  = lambda x: (rodriguesResidual(K1,M1,p1,K2,p2,x)**2).sum()
	#rod_func  = lambda x: rodriguesResidual(K1, M1, p1, K2, p2, x)

	#x_star, flag = leastsq(rod_func, x_init)
	res = minimize(rod_func, x_init)
	x_star = res.x
	P_star  = x_star[:-6].reshape(N, 3)
	r2_star = x_star[-6:-3]
	R2_star = rodrigues(r2_star)
	t2_star = x_star[-3:].reshape(-1, 1)
	M2 = np.hstack((R2_star, t2_star))

	return M2, P_star 

if __name__ == '__main__':
	data = np.load('../data/some_corresp.npz')
	pts1, pts2 = data['pts1'], data['pts2']
	M = 400
	eightpoint(pts1, pts2, M)
