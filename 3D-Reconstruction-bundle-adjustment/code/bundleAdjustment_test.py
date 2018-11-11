from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import numpy as np
import submission as sub


corresp    = np.load('../data/some_corresp.npz')
cameraMats = np.load('q3_3.npz')
intrinsics = np.load('../data/intrinsics.npz')
p1 = corresp['pts1']
p2 = corresp['pts2']
K1 =  intrinsics['K1']
K2 =  intrinsics['K2']

M1 = np.vstack((np.eye(3), np.zeros(3))).T
C1 = K1 @ M1
M2_init, C2, P_init  = cameraMats['M2'], cameraMats['C2'] , cameraMats['P']

fig1 = plt.figure()
ax = fig1.add_subplot(111, projection='3d')
ax.scatter(P_init[:, 0], P_init[:, 1], P_init[:, 2], c='b', marker='o')



plt.show()
