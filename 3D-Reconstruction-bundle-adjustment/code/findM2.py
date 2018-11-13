'''
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, p1, p2, R and P to q3_3.mat
'''
import numpy as np
import submission as sub
import helper
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

corresp = np.load('../data/some_corresp.npz')
intrinsics = np.load('../data/intrinsics.npz')

K1 =  intrinsics['K1']
K2 =  intrinsics['K2']
pts1 = corresp['pts1']
pts2 = corresp['pts2']
scale = 640


F = sub.eightpoint(pts1, pts1, scale)
E = sub.essentialMatrix(F, K1, K2)
M1 = np.vstack((np.eye(3), np.zeros(3))).T
M2s = helper.camera2(E)

C1 = K1 @ M1

best_M2 = None
P_best  = None
C2_best = None
best_valid = 0

for i in range(M2s.shape[2]):

	C2 = K2 @ M2s[:, :, i]
	P, error = sub.triangulate(C1, pts1, C2, pts2)	
	valid_pts = np.sum(P[:, 2] > 0)
	print('valid_pts: ', valid_pts, error)
	fig1 = plt.figure()
	ax = fig1.add_subplot(111, projection='3d')
	ax.scatter(P[:, 0], P[:, 1], P[:, 2], c='b', marker='o', s = 1)
	plt.show()	
	if valid_pts > best_valid:
		
		best_M2 = M2s[:, :, i]
		best_error = error
		best_valid = valid_pts
		P_best  = P
		C2_best = C2
		print('new best found', i) 
			

fig1 = plt.figure()
ax = fig1.add_subplot(111, projection='3d')
ax.scatter(P_best[:, 0], P_best[:, 1], P_best[:, 2], c='b', marker='o', s = 1)
plt.show()		

np.savez('q3_3.npz',M2=best_M2, C2=C2_best, P=P_best)
print('Find M2 dumping done !!')

