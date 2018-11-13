import numpy as np
import submission as sub
import helper
import cv2
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

noise_data = np.load('../data/some_corresp_noisy.npz')

pts1, pts2 = noise_data['pts1'], noise_data['pts2']

im1 = plt.imread('../data/im1.png')
im2 = plt.imread('../data/im2.png')

N = pts1.shape[0]

M = 640

#F8 = sub.eightpoint(data['pts1'], data['pts2'], M)


inliers = np.load('best_inliers.npy').reshape([-1])
pts1_in, pts2_in = pts1[inliers, :], pts2[inliers, :]
p = pts1_in.shape[0]
pts1_h = np.hstack((pts1_in, np.ones((p, 1))))
pts2_h = np.hstack((pts2_in, np.ones((p, 1))))

bestFs = sub.sevenpoint(pts1_in, pts2_in, M)
tol = 0.001
bestF_inlier_count = 0

for F in bestFs:

	dst = np.diag(pts2_h @ F @ pts1_h.T)
	inliers =  np.abs(dst) < tol
	inliers_count = np.sum(inliers)

	if inliers_count >  bestF_inlier_count:
		#print('new max: ', inliers_count)
		bestF_inlier_count = inliers_count
		bestF = F
		bestF_inliers = inliers

bestF = helper.refineF(bestF, pts1_in, pts2_in)

im1_temp = im1.copy()
im2_temp = im2.copy()

'''
for i in range(pts1_in.shape[0]):
	x1, y1 = pts1_in[i, 0], pts1_in[i, 1]
	
	x2, y2 = sub.epipolarCorrespondence(im1_temp, im2_temp, bestF, x1, y1)
	cv2.circle(im1_temp, (x1, y1), 5, (0,0,255), -1)
	cv2.circle(im2_temp, (x2, y2), 5, (0,0,255), -1)
	cv2.imshow('input ', im1_temp)
	cv2.imshow('output ', im2_temp)
	cv2.waitKey(10)
	
cv2.waitKey(30000)
cv2.destroyAllWindows()
'''
intrinsics = np.load('../data/intrinsics.npz')

K1 =  intrinsics['K1']
K2 =  intrinsics['K2']
		
E = sub.essentialMatrix(bestF, K1, K2)
M1 = np.vstack((np.eye(3), np.zeros(3))).T
M2s = helper.camera2(E)

C1 = K1 @ M1

best_M2 = None
P_best  = None
C2_best = None
best_error = np.inf

for i in range(4):

	C2 = K2 @ M2s[:, :, i]
	P, error = sub.triangulate(C1, pts1_in, C2, pts2_in)
	fig1 = plt.figure()
	ax = fig1.add_subplot(111, projection='3d')
	ax.scatter(P[:, 0], P[:, 1], P[:, 2], c='b', marker='o')
	plt.show()		
	print('error: ', error)
	if error <= best_error:
		
		best_M2 = M2s[:, :, i]
		best_error = error
		P_best  = P
		C2_best = C2
		print('new best found', ) 

print('best error: ', best_error, best_error/pts1_in.shape[0])

#fig1 = plt.figure()
#ax = fig1.add_subplot(111, projection='3d')
#ax.scatter(P_best[:, 0], P_best[:, 1], P_best[:, 2], c='b', marker='o')
#plt.show()
