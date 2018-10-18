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
    
    #mask = np.ones(image1.shape, dtype=bool)
    mask = image2 - image1_w
    
    return mask
