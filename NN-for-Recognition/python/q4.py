import numpy as np

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.filters
import skimage.morphology
import skimage.segmentation


# takes a color image
# returns a list of bounding boxes and black_and_white image
def findLetters(image):
	bboxes = []
	bw = None
	# insert processing in here
	# one idea estimate noise -> denoise -> greyscale -> threshold -> morphology -> label -> skip small boxes 
	# this can be 10 to 15 lines of code using skimage functions
	#cv2.imshow('input image', image)

	image = skimage.restoration.denoise_bilateral(image, multichannel=True)
	gray  = skimage.color.rgb2grey(image)

	thresh_min = skimage.filters.threshold_minimum(gray)
	binary_im = gray < thresh_min
	binary_im = skimage.morphology.closing(binary_im)
	bw = skimage.measure.label(binary_im)

	for region in skimage.measure.regionprops(bw):
		
		if region.area >= 200:
			bboxes.append(region.bbox)

	#cv2.imshow('denoised image', label_image)
	#cv2.waitKey(10000)
	#cv2.destroyAllWindows()
	return bboxes, bw.astype('float')

