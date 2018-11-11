"""
Check the dimensions of function arguments
This is *not* a correctness check

Written by Chen Kong, 2018.
"""
import numpy as np
import matplotlib.pyplot as plt
import submission as sub
import helper

data = np.load('../data/some_corresp.npz')
noise_data = np.load('../data/some_corresp_noisy.npz')

im1 = plt.imread('../data/im1.png')
im2 = plt.imread('../data/im2.png')

N = data['pts1'].shape[0]
M = 640
'''
# 2.1
F8 = sub.eightpoint(noise_data['pts1'], noise_data['pts2'], M)
assert F8.shape == (3, 3), 'eightpoint returns 3x3 matrix'

print(F8)
test_pt = np.hstack([data['pts1'][0], 1])
I = F8 @ test_pt
print('I: ', (I / I[-1]), 'test_pt: ', test_pt)
helper.displayEpipolarF(im1, im2, F8)

# 2.2

randPtsIdx = np.array([14, 98, 47, 16, 92, 56, 54])#
#randPtsIdx = np.random.choice(N, 7, replace=False)
print('pts: ', randPtsIdx)

pts1 = data['pts1'][randPtsIdx, :]
pts2 = data['pts2'][randPtsIdx, :]
M = max(np.max(pts1),  np.max(pts2))
F7 = sub.sevenpoint(pts1, pts2, M)

assert len(F7) == 1 | len(F7) == 3, 'sevenpoint returns length-1/3 list'

for f7 in F7:
	assert f7.shape == (3, 3), 'seven returns list of 3x3 matrix'
	
np.savez('q2_2.npz', F=F7[0],
					 M=M, 
					 pts1=pts1,
					 pts2=pts2)
print('sevenpoint dumping done !!')

helper.displayEpipolarF(im1, im2, F7[0])

# 3.1
C1 = np.concatenate([np.random.rand(3, 3), np.ones([3, 1])], axis=1)
C2 = np.concatenate([np.random.rand(3, 3), np.ones([3, 1])], axis=1)

P, err = sub.triangulate(C1, data['pts1'], C2, data['pts2']);
assert P.shape == (N, 3), 'triangulate returns Nx3 matrix P'
assert np.isscalar(err), 'triangulate returns scalar err'

# 4.1
x2, y2 = sub.epipolarCorrespondence(im1, im2, F8, data['pts1'][0, 0], data['pts1'][0, 1])
assert np.isscalar(x2) & np.isscalar(y2), 'epipolarCoorespondence returns x & y coordinates'
helper.epipolarMatchGUI(im1, im2, F8)

# 5.1


F, inliers = sub.ransacF(noise_data['pts1'], noise_data['pts2'], M);
assert F.shape == (3, 3), 'ransacF returns 3x3 matrix'
print('It was noisy')
helper.displayEpipolarF(im1, im2, F)

'''
# 5.2

r =np.ones([3, 1])
print('r orig: ', r)
R = sub.rodrigues(r)
print('R: ', R)
assert R.shape == (3, 3), 'rodrigues returns 3x3 matrix'

R = np.eye(3)
r = sub.invRodrigues(R)
print('r: ', r)
assert (r.shape == (3, )) | (r.shape == (3, 1)), 'invRodrigues returns 3x1 vector'
'''
# 5.3

#P  = np.random.randint(0, 25, (N, 3))
K1 = np.random.rand(3, 3)
K2 = np.random.rand(3, 3)
M1 = np.concatenate([np.random.rand(3, 3), np.ones([3, 1])], axis=1)
M2 = np.concatenate([np.random.rand(3, 3), np.ones([3, 1])], axis=1)
r2 = np.ones(3)
t2 = np.ones(3)
x = np.concatenate([P.reshape([-1]), r2, t2])

#############TODO###########MODIFIED
residuals = sub.rodriguesResidual(x, K1, M1, data['pts1'], K2, data['pts1'])
assert residuals.shape == (4 * N, ), 'rodriguesResidual returns vector of size 4Nx1'
print(residuals)

M2, P = sub.bundleAdjustment(K1, M1, data['pts1'], K2, M2, data['pts1'], P);
assert M2.shape == (3, 4), 'bundleAdjustment returns 3x4 matrix M'
assert P.shape == (N, 3), 'bundleAdjustment returns Nx3 matrix P'

print(M2, P)
print('Format check passed.')
'''
