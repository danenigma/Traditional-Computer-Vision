import cv2
import os
import numpy as np
import BRIEF
import planarH
import keypointDetect
import panoramas
import augmentedReality
import matplotlib.pyplot as plt
def test_keypoint():
	levels = [-1,0,1,2,3,4]
	im = cv2.imread('../data/model_chickenbroth.jpg')
	#im = cv2.imread('../data/prince_book.jpeg')
	locsDoG, gaussian_pyramid = keypointDetect.DoGdetector(im)
	
	N, _ = locsDoG.shape

	print('N:', N)

	im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	plt.imshow(im, cmap='gray')
	for i in range(N):
		plt.plot([locsDoG[i][0]],[locsDoG[i][1]],'ro')
	plt.show()
def test_BRIEF():
	
	im1 = cv2.imread('../data/model_chickenbroth.jpg')
	im2 = cv2.imread('../data/chickenbroth_05.jpg')

	locs1, desc1 = BRIEF.briefLite(im1)
	locs2, desc2 = BRIEF.briefLite(im2)
	matches = BRIEF.briefMatch(desc1, desc2)
	BRIEF.plotMatches(im1,im2,matches,locs1,locs2)

def test_panorama():

	im1 = cv2.imread('../data/incline_L.png')
	im2 = cv2.imread('../data/incline_R.png')
	
	H_file = '../results/q6_1.npy'
	if os.path.isfile(H_file):
		H2to1= np.load(H_file)
	else:
		print('RANSACing....')
		locs1, desc1 = BRIEF.briefLite(im1)
		locs2, desc2 = BRIEF.briefLite(im2)
		matches = BRIEF.briefMatch(desc1, desc2)
		H2to1 = planarH.ransacH(matches, locs1, locs2, num_iter=5000, tol=2)
		np.save('../results/q6_1.npy', H2to1)

	im3 = panoramas.imageStitching(im1, im2, H2to1)
	cv2.imwrite('../results/6_1.jpg', im3)
	cv2.imshow('panoramas', im3)
	cv2.waitKey(5000)
	cv2.destroyAllWindows()
def test_panorama_noclip():

	im1 = cv2.imread('../data/incline_L.png')
	im2 = cv2.imread('../data/incline_R.png')
	
	H_file = '../results/q6_1.npy'
	if os.path.isfile(H_file):
		H2to1= np.load(H_file)
	else:
		print('RANSACing....')
		locs1, desc1 = BRIEF.briefLite(im1)
		locs2, desc2 = BRIEF.briefLite(im2)
		matches = BRIEF.briefMatch(desc1, desc2)
		H2to1 = planarH.ransacH(matches, locs1, locs2, num_iter=5000, tol=2)
		print('dumping npy file...')
		np.save('../results/q6_1.npy', H2to1)

	pano_im = panoramas.imageStitching_noClip(im1, im2, H2to1)
	cv2.imwrite('../results/q6_2_pan.jpg', pano_im)
	cv2.imshow('panoramas', pano_im)
	cv2.waitKey(10000)
	cv2.destroyAllWindows()

def test_generate_panorama():
	im1 = cv2.imread('../data/incline_L.png')
	im2 = cv2.imread('../data/incline_R.png')
	pano_im = panoramas.generatePanorama(im1, im2)
	cv2.imwrite('../results/q6_3.jpg', pano_im)
	cv2.imshow('panoramas', pano_im)
	cv2.waitKey(10000)
	cv2.destroyAllWindows()

def test_augmented_reality():
	augmentedReality.project_sphere()
	

if __name__ == '__main__':
	#test_keypoint()
	#test_BRIEF()
	#test_panorama()	
	test_panorama_noclip()	
	#test_generate_panorama()
	#test_augmented_reality()
