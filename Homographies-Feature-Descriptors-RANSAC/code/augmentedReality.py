import numpy as np
import csv
import cv2
import math
import planarH
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def getSphere():

	reader = csv.reader(open("../data/sphere.txt"), delimiter="\t")
	count = 0
	i = 0
	sphere = np.zeros((3, 961))
	for row in reader:
		r  = row[0].split('  ')
		sphere[i, :] = np.array([float(j) for j in r[1:]])
		i+=1
	return sphere
	
def compute_extrinsics(K, H):
	H_prim = np.matmul(np.linalg.inv(K), H)
	
	[U, S, V] = np.linalg.svd(H_prim[:,:2], True)

	L = np.array([[1, 0],
				  [0, 1],
				  [0, 0]])

	R = np.zeros((3,3))
	R12  = np.matmul(U, np.matmul(L, V))
	R3   = np.cross(R12[:, 0], R12[:, 1])
	R[:,:2] = R12
	R[:, 2] = R3
	
	
	if np.linalg.det(R) == -1:
		R[:, 2] = R[:, 2]*-1	
	lamda  = np.sum(np.divide(H_prim[:,:2], R[:, :2]))/6
	t = H_prim[:, 2]/lamda
	return R, t
	
def project_extrinsics(K, W, R, t):
	R_t  = np.zeros((3, 4))
	R_t[:, :3] = R
	R_t[:, 3]  = t
	proj_mat = np.dot(K, R_t)
	Xout = cv2.perspectiveTransform(W.T.reshape(-1, 1 , 3), proj_mat).T
	return Xout.squeeze(1)
	
if __name__ =='__main__':


	W = np.array([[0., 18.2, 18.2, 0.],
				  [0., 0., 26., 26.],
				  [0., 0., 0., 0.]]).astype('float')
	X = np.array([[483, 1704, 2175, 67],
				  [810, 781, 2217, 2286]])
	K = np.array([[3043.72, 0.0, 1196.00],
				  [0.0, 3043.72, 1604.00],
				  [0.0, 0.0, 1.0]])
				  
	M = np.array([[1,   0., 300.],
				  [0.,  1,  600],
				  [0.,  0., 1.]]).astype('float')			  
				  
	H = planarH.computeH(X, W[:2, :])
	R, t = compute_extrinsics(K, H)
	sphere = getSphere()

	#plot actual sphere
	#fig = plt.figure()
	#ax = plt.axes(projection='3d')
	#ax.scatter3D(sphere[0,:], sphere[1,:], sphere[2,:], c=sphere[2,:], cmap='Greens');
	#plt.show()

	projectedSphere = project_extrinsics(K, sphere, R, t)
	projectedSphereH = np.ones((3, projectedSphere.shape[1]))
	projectedSphereH[:2, :] = projectedSphere
	
	shiftedSphere = np.matmul(M, projectedSphereH)[:2, :]
	N  = shiftedSphere.shape[1]
	im = cv2.imread('../data/prince_book.jpeg')
	plt.imshow(im, cmap='gray')
	
	for i in range(N):
		plt.plot(shiftedSphere[0, i], shiftedSphere[1, i], 'yo')	
	plt.show()
	

