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
print('M2 before: ', M2)
M2, P_star = sub.bundleAdjustment(K1, M1, pts1, K2, M2, pts2, P_best)
print('bundle adjustment done ....')
print('M2 after: ', M2)
C2 = K2 @ M2
P, error = sub.triangulate(C1, pts1, C2, pts2)
print('after bundle: ', error)
fig2 = plt.figure()
ax = fig2.add_subplot(111, projection='3d')
ax.scatter(P_star[:, 0], P_star[:, 1], P_star[:, 2], c='g', marker='o', s=1)
ax.scatter(P_best[:, 0], P_best[:, 1], P_best[:, 2], c='b', marker='o', s = 1)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()

'''	
#p1_data = np.load('../data/templeCoords.npz')

X1, Y1 = pts1[:, 0].reshape(-1, 1), pts1[:, 1].reshape(-1, 1)#p1_data['x1'], p1_data['y1']
X2 = []
Y2 = []
print('bestF: ', bestF)
print('X1: ', X1)
for x1, y1 in zip(X1, Y1):
	[x2, y2] = sub.epipolarCorrespondence(im1, im2, bestF, x1[0], y1[0])
	X2.append(x2)
	Y2.append(y2)

X2, Y2 = np.array(X2), np.array(Y2)
p1 = np.hstack((X1.reshape(-1, 1), Y1.reshape(-1, 1)))
p2 = np.hstack((X2.reshape(-1, 1), Y2.reshape(-1, 1)))
P, error =  sub.triangulate(C1, p1, C2_best, p2)

print('reprojection error before bundle: ', error)

fig1 = plt.figure()
ax = fig1.add_subplot(111, projection='3d')
ax.scatter(P[:, 0], P[:, 1], P[:, 2], c='b', marker='o')

print('bundle adjustment running....')
M2, P_star = sub.bundleAdjustment(K1, M1, p1, K2, best_M2, p2, P)

print('Fstar: ', M2[:, :3])
print('P stat: ', np.max(P_star), np.min(P_star), np.median(P_star))
#P_star[np.abs(P_star)>1] = 0

print('bundle adjustment done ....')

fig2 = plt.figure()
ax = fig2.add_subplot(111, projection='3d')

ax.scatter(P_star[:, 0], P_star[:, 1], P_star[:, 2], c='g', marker='o')
C2_star = K2 @ M2 

P, error =  sub.triangulate(C1, p1, C2_star, p2)
print('reprojection error after bundle: ', error)

plt.show()

'''








