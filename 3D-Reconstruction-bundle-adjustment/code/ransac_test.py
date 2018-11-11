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

pts1, pts2 = noise_data['pts1'], noise_data['pts2']

p = pts1.shape[0]
pts1_h = np.hstack((pts1, np.ones((p, 1))))
pts2_h = np.hstack((pts2, np.ones((p, 1))))



#bestF, inliers = sub.ransacF(noise_data['pts1'], noise_data['pts2'], M);
#np.save('inliers4.npy', inliers)
#print('Done!!')


inliers = np.load('best_inliers.npy').reshape([-1])

bestFs = sub.sevenpoint(pts1[inliers, :], pts2[inliers, :], M)
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

bestF = helper.refineF(bestF, pts1[bestF_inliers, :], pts2[bestF_inliers, :])
	
helper.displayEpipolarF(im1, im2, bestF)














