'''
Q4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter3
'''
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import submission as sub
import helper

data = np.load('../data/some_corresp.npz')
im1 = plt.imread('../data/im1.png')
im2 = plt.imread('../data/im2.png')

N = data['pts1'].shape[0]
M = 640

F8 = sub.eightpoint(data['pts1'], data['pts2'], M)

p1_data = np.load('../data/templeCoords.npz')

X1, Y1 = p1_data['x1'], p1_data['y1']
 
intrinsics = np.load('../data/intrinsics.npz')
K1 =  intrinsics['K1']
K2 =  intrinsics['K2']


M1 = np.vstack((np.eye(3), np.zeros(3))).T
C1 = K1 @ M1

X2 = []
Y2 = []

for x1, y1 in zip(X1, Y1):

	[x2, y2] = sub.epipolarCorrespondence(im1, im2, F8, x1[0], y1[0])
	X2.append(x2)
	Y2.append(y2)
	
X2, Y2 = np.array(X2), np.array(Y2)


p1 = np.hstack((X1.reshape(-1, 1), Y1.reshape(-1, 1)))
p2 = np.hstack((X2.reshape(-1, 1), Y2.reshape(-1, 1)))
E = sub.essentialMatrix(F8, K1, K2)
M2s = helper.camera2(E)
best_valid_pts = 0

P_best = None
M2 = None
for i in range(4):

	C2 = K2 @ M2s[:, :, i]
	P, error = sub.triangulate(C1, p1, C2, p2)
	valid_pts = np.sum(P[:, 2] > 0)
	print('valid_pts: ', valid_pts)
	
	if valid_pts > best_valid_pts:
		print('found new M2')
		best_valid_pts = valid_pts 	
		P_best = P
		M2 = M2s[:, :, i]
print(M2)			
fig1 = plt.figure()
ax = fig1.add_subplot(111, projection='3d')
ax.scatter(P_best[:, 0], P_best[:, 1], P_best[:, 2], c='b', marker='o', s = 1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()		

C2 = K2 @ M2
P, error = sub.triangulate(C1, p1, C2, p2)
print('before bundle: ', error)
print('Running bundle adjustment ....')
M2, P_star = sub.bundleAdjustment(K1, M1, p1, K2, M2, p2, P_best)
print('bundle adjustment done ....')
C2 = K2 @ M2
P, error = sub.triangulate(C1, p1, C2, p2)
print('after bundle: ', error)
fig2 = plt.figure()
ax = fig2.add_subplot(111, projection='3d')
ax.scatter(P_star[:, 0], P_star[:, 1], P_star[:, 2], c='g', marker='o', s=1)

plt.show()


