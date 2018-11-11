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

cameraMats = np.load('q3_3.npz')

M1 = np.vstack((np.eye(3), np.zeros(3))).T
C1 = K1 @ M1
C2 = cameraMats['C2']
M2_init = cameraMats['M2']
X2 = []
Y2 = []

for x1, y1 in zip(X1, Y1):

	[x2, y2] = sub.epipolarCorrespondence(im1, im2, F8, x1[0], y1[0])
	X2.append(x2)
	Y2.append(y2)
	
X2, Y2 = np.array(X2), np.array(Y2)
p1 = np.hstack((X1.reshape(-1, 1), Y1.reshape(-1, 1)))
p2 = np.hstack((X2.reshape(-1, 1), Y2.reshape(-1, 1)))

P, error =  sub.triangulate(C1, p1, C2, p2)

print(P.shape)
fig1 = plt.figure()
ax = fig1.add_subplot(111, projection='3d')
ax.scatter(P[:, 0], P[:, 1], P[:, 2], c='b', marker='o')
print('Running bundle adjustment ....')
M2, P_star = sub.bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P)
print('M2 init: ', M2_init)
print('M2: ', M2)
print('bundle adjustment done ....')
fig2 = plt.figure()
ax = fig2.add_subplot(111, projection='3d')
ax.scatter(P_star[:, 0], P_star[:, 1], P_star[:, 2], c='g', marker='o')


plt.show()


