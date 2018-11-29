import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation

from nn import *
from q4 import *
# do not include any more libraries here!
# no opencv, no sklearn, etc!

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

for img in os.listdir('../images'):
	im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images',img)))
	bboxes, bw = findLetters(im1)
	print('finding letters for: ', img)
	plt.imshow(bw)
	
	for bbox in bboxes:
		minr, minc, maxr, maxc = bbox
		rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
		                        fill=False, edgecolor='red', linewidth=2)
		plt.gca().add_patch(rect)
	plt.show()
	
	bboxes_arr = np.array(bboxes)
	
	row_data = np.array(bboxes)[:, 0].reshape(-1, 1)
	rows = []
	row = []
	
	for i in range(row_data.shape[0]-1):

		#print(crop_im.shape)
		row.append(bboxes[i])
		if 	row_data[i+1] - row_data[i] > 100:
			rows.append(row)
			row = []	 
	row.append(bboxes[-1])
	rows.append(row)

	import pickle
	import string
	letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
	params = pickle.load(open('q3_weights.pickle','rb'))

	for row in rows:
		sorted_row = np.array(row)[np.argsort(np.array(row)[:,1]), :]
		row_letters = []
		row_X = []
		for bbox in sorted_row:
			
		
			crop_im = bw[bbox[0]:bbox[2], bbox[1]:bbox[3]]
			#plt.subplot(2,1,1)
			#plt.imshow(crop_im)

			H, W = crop_im.shape
			size  = max(H, W)
			h_pad  = (size - H)//2 + int(size*0.2)
			w_pad  = (size - W)//2 + int(size*0.2)
			crop_im_pad = np.pad(crop_im, 
								[(h_pad, h_pad),(w_pad, w_pad)], 
								'constant',
								constant_values=(crop_im[0, 0], crop_im[0, 0]))					
			crop_im_final = skimage.transform.resize(crop_im_pad, 
													(32, 32))		
			crop_im_final = 0.95*skimage.filters.gaussian(crop_im_final, sigma=0.5)
			#print(crop_im_final.min(), crop_im_final.max())
			row_X.append(crop_im_final.T.flatten())
			#plt.subplot(2,1,2)			
			#print(crop_im_final)
			#plt.imshow(crop_im_final)
			#plt.show()
			
		row_X = np.array(row_X)
		h1 = forward(row_X, params,'layer1')
		probs = forward(h1, params, 'output', softmax)

		row_txt = [letters[i] for i in np.argmax(probs, axis=1)]
		print(''.join(row_txt))
	
	
	
	
	'''
    # find the rows using..RANSAC, counting, clustering, etc.
    
    # crop the bounding boxes
    # note.. before you flatten, transpose the image (that's how the dataset is!)
    # consider doing a square crop, and even using np.pad() to get your images looking more like the dataset
    
    # load the weights
    # run the crops through your neural network and print them out
    import pickle
    import string
    letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
    params = pickle.load(open('q3_weights.pickle','rb'))
	
	'''   
	
