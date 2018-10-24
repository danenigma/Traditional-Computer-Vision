import numpy as np
import cv2
from LucasKanadeAffine import LucasKanadeAffine
from scipy.ndimage import affine_transform
from InverseCompositionAffine import InverseCompositionAffine
def SubtractDominantMotion(image1, image2):
	# Input:
	#	Images at time t and t+1 
	# Output:
	#	mask: [nxm]
	# put your implementation here

	M = LucasKanadeAffine(image1, image2)
	
	#M = InverseCompositionAffine(image1, image2)

	M_temp = np.eye(3,3)
	M_temp[:2, :] = M
	M_inv = np.linalg.inv(M_temp)
	H, W = image2.shape

	image1_w = affine_transform(image1,  M_inv[:2,:2], offset =[M_inv[0, 2], M_inv[1, 2]])


	retval, im2_mask = cv2.threshold(image1_w, 0, 1, cv2.THRESH_BINARY)
	mask_gray = cv2.absdiff(image2*im2_mask, image1_w)
	#print(np.max(mask_gray), np.min(mask_gray))
	
	simple_mask_gray = cv2.absdiff(image2, image1)
	retval, mask = cv2.threshold(mask_gray.copy(), 0.2, 1, cv2.THRESH_BINARY)
	kernel = np.ones((5,5),np.uint8)
	
	mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)),iterations = 3)
	mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)),iterations = 1)
	
	
	retval, simple_mask = cv2.threshold(simple_mask_gray, 0.2, 1, cv2.THRESH_BINARY)
	simple_mask = cv2.dilate(simple_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)),iterations = 3)
	simple_mask = cv2.erode(simple_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)),iterations = 1)

	#mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
	#cv2.imshow('simple_mask', simple_mask)
	
	return mask
