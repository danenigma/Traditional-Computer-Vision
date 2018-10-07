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
def rotate_image(mat, angle):
    height, width = mat.shape[:2]
    image_center = (width / 2, height / 2)

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1)

    radians = math.radians(angle)
    sin = math.sin(radians)
    cos = math.cos(radians)
    bound_w = int((height * abs(sin)) + (width * abs(cos)))
    bound_h = int((height * abs(cos)) + (width * abs(sin)))

    rotation_mat[0, 2] += ((bound_w / 2) - image_center[0])
    rotation_mat[1, 2] += ((bound_h / 2) - image_center[1])

    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat 

if __name__ == '__main__':

	im1 = cv2.imread('../data/model_chickenbroth.jpg')
	locs1, desc1 = briefLite(im1)
	bar_graph = []

	for angle in range(0, 370, 10):
	
		im_r = rotate_image(im1, angle)
		locs2, desc2 = briefLite(im_r)
		matches = briefMatch(desc1, desc2)
		bar_graph.append(len(matches))
		print(angle , len(matches))
		
	plt.bar(np.arange(0, 370, 10), np.array(bar_graph))
	plt.xlabel('Angle in Degrees')
	plt.ylabel('Match count')
	plt.show()


