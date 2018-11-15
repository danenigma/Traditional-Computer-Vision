import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import submission as sub
import helper

data = np.load('../data/some_corresp.npz')
noise_data = np.load('../data/some_corresp_noisy.npz')

im1 = plt.imread('../data/im1.png')
im2 = plt.imread('../data/im2.png')

N = data['pts1'].shape[0]
M = 640

pts1, pts2 = noise_data['pts1'], noise_data['pts2']




#bestF, inliers = sub.ransacF(noise_data['pts1'], noise_data['pts2'], M);
#np.save('inliers4.npy', inliers)
#print('Done!!')


inliers = np.load('best_inliers.npy').reshape([-1])
pts1, pts2 = pts1[inliers, :], pts2[inliers, :]

p = pts1.shape[0]
pts1_h = np.hstack((pts1, np.ones((p, 1))))
pts2_h = np.hstack((pts2, np.ones((p, 1))))

bestFs = sub.sevenpoint(pts1, pts2, M)
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

bestF = helper.refineF(bestF, pts1, pts2)

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

best_valid_pts = 0
P_best = None
M2 = None

print('inlier count: ', inliers.shape)
for i in range(4):

	C2 = K2 @ M2s[:, :, i]
	P, error = sub.triangulate(C1, pts1, C2, pts2)	
	valid_pts = np.sum(P[:, 2] > 0)
	print('valid_pts: ', valid_pts)
	if valid_pts > best_valid_pts:
		print('new best found', i) 
		valid_pts = best_valid_pts
		M2 = M2s[:, :, i]
		P_best = P
		
fig1 = plt.figure()
ax = fig1.add_subplot(111, projection='3d')
ax.scatter(P_best[:, 0], P_best[:, 1], P_best[:, 2], c='b', marker='o', s = 1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

C2 = K2 @ M2
P, error = sub.triangulate(C1, pts1, C2, pts2)
print('before bundle: ', error)
print('Running bundle adjustment ....')
M2, P_star = sub.bundleAdjustment(K1, M1, pts1, K2, M2, pts2, P_best)
print('bundle adjustment done ....')
C2 = K2 @ M2
P, error = sub.triangulate(C1, pts1, C2, pts2)
print('after bundle: ', error)
fig2 = plt.figure()
ax = fig2.add_subplot(111, projection='3d')
ax.scatter(P_star[:, 0], P_star[:, 1], P_star[:, 2], c='g', marker='o')
ax.scatter(P_best[:, 0], P_best[:, 1], P_best[:, 2], c='r', marker='o')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()








