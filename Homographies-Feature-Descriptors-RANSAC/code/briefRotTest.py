from BRIEF import briefLite, briefMatch, plotMatches
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

def drawKeys(im, locsDoG):
	N, _ = locsDoG.shape
	for i in range(N):
		cv2.circle(im, (locsDoG[i][0], locsDoG[i][1]), 1, (0,0,255), 1)
	return im
	
def rotateImage(im, angle):

	H, W = im.shape[:2]
	im_center  = (W//2, H//2)
	rot_matrix = cv2.getRotationMatrix2D(im_center, angle, 1)
	corner  = np.matmul(np.abs(rot_matrix[:, :2]), np.array([W, H]).T).astype('int')
	rot_matrix[0, 2] += ((corner[0] / 2) - im_center[0])
	rot_matrix[1, 2] += ((corner[1] / 2) - im_center[1])
	im_out = cv2.warpAffine(im, rot_matrix, (corner[0], corner[1]))
	
	return im_out 

if __name__ == '__main__':

	im1 = cv2.imread('../data/model_chickenbroth.jpg')
	locs1, desc1 = briefLite(im1)
	bar_graph = []

	for angle in range(0, 370, 10):
	
		im_r = rotateImage(im1, angle)
		locs2, desc2 = briefLite(im_r)
		matches = briefMatch(desc1, desc2)
		bar_graph.append(len(matches))
		
	plt.bar(np.arange(0, 370, 10), np.array(bar_graph))
	plt.xlabel('Angle in Degrees')
	plt.ylabel('Match count')
	plt.show()


