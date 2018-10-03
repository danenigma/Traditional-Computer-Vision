import numpy as np
import csv

def compute_extrinsics(K, H):
	H_prim = np.matmul(np.linalg.inv(K), H)
	[U, S, V] = np.linalg.svd(H_prim[:,:2], False)
	L = np.array([[1, 0],
				  [0, 1]])

	R  = np.matmul(U, np.matmul(L, V))
	R3 = np.cross(R[:, 0], R[:, 1])
	lamda  = np.sum(np.divide(H_prim[:,:2], R))/6
	t = H_prim[:, 2]/lamda
	return R, t
	
def project_extrinsics(K, W, R, t):
	
	return X

W = np.array([[0., 18.2, 18.2, 0.],
			  [0., 0., 26., 26.],
			  [0., 0., 0., 0.]]).astype('float')
X = np.array([[483, 1704, 2175, 67],
			  [810, 781, 2217, 2286]])
K = np.array([[3043.72, 0.0, 1196.00],
			  [0.0, 3043.72, 1604.00],
			  [0.0, 0.0, 1.0]])
			  
H = np.random.randn(3,3)
R, t = compute_extrinsics(K, H)
print(R)
print(t)
reader = csv.reader(open("../data/sphere.txt"), delimiter=" ")

for row in reader:
	print(row[0])
	
