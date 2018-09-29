import numpy as np
import multiprocessing
import imageio
import scipy.ndimage
import skimage.color
import sklearn.cluster
import scipy.spatial.distance
import os,time
import matplotlib.pyplot as plt
import util
import random



def image_as_float(image):
	return image.astype('float')/image.max()
	
def extract_filter_responses(image):
	'''
	Extracts the filter responses for the given image.

	[input]
	* image: numpy.ndarray of shape (H,W) or (H,W,3)
	[output]
	* filter_responses: numpy.ndarray of shape (H,W,3F)
	'''
	
	# ----- TODO -----
	# if gray scale
	if len(image.shape)==2:
		image = np.stack((image,)*3, -1)
	# if 4 channel image take the first 3
	elif(image.shape[2]==4):
		image = image[:, :, :3]	
	
	image = skimage.color.rgb2lab(image_as_float(image))
	
	H, W, num_channels = image.shape

	F = 20
	dim_counter  = 0 
	num_channels = 3

	scales =  [1, 2, 4, 8, 8*np.sqrt(2)]
	
	

	filter_responses = np.zeros((H, W, 3*F)).astype(np.float64)
	
		
	for sc in scales:
	
		for ch in range(num_channels):	
		    # gaussian 	
			filter_responses[:, :, dim_counter] = scipy.ndimage.gaussian_filter(
																 image[:, :, ch],
																 [sc, sc])

			dim_counter += 1
			
		for ch in range(num_channels):
			#laplacian
			filter_responses[:, :, dim_counter] = scipy.ndimage.gaussian_laplace(
																 image[:, :, ch],
																 [sc, sc])
			dim_counter += 1
			
		for ch in range(num_channels):
			# gaussian derivative in y 
			filter_responses[:, :, dim_counter] = scipy.ndimage.gaussian_filter(
																 image[:, :, ch],
																 [sc, sc], [0, 1])
		 
			dim_counter += 1

		for ch in range(num_channels):
			# gaussian derivative in x
			filter_responses[:, :, dim_counter] = scipy.ndimage.gaussian_filter(
																 image[:, :, ch],
																 [sc, sc], [1, 0])
																 
			dim_counter += 1
			
	return filter_responses




def get_visual_words(image, dictionary):
	'''
	Compute visual words mapping for the given image using the dictionary of visual words.

	[input]
	* image: numpy.ndarray of shape (H,W) or (H,W,3)
	
	[output]
	* wordmap: numpy.ndarray of shape (H,W)
	'''
	
	# ----- TODO -----
	
	f_responses = extract_filter_responses(image)
	H, W, F_3 = f_responses.shape
	f_responses_mat = f_responses.reshape(-1, F_3)
	distances = scipy.spatial.distance.cdist(f_responses_mat, dictionary)
	wordmap = np.argmin(distances, axis=1)
 
	return wordmap.reshape(H, W)
	


def compute_dictionary_one_image(args):
	'''
	Extracts random samples of the dictionary entries from an image.
	This is a function run by a subprocess.

	[input]
	* i: index of training image
	* alpha: number of random samples
	* image_path: path of image file
	* time_start: time stamp of start time

	[saved]
	* sampled_response: numpy.ndarray of shape (alpha,3F)
	'''


	i, alpha, image_path, time_start = args

	# ----- TODO -----
	print('Processing Image: ', i)
	
	image = imageio.imread(image_path) 
	
	response_maps = extract_filter_responses(image) 
	H, W, num_channels = response_maps.shape
	
	rand_xs  = np.random.randint(0, H, alpha)
	rand_ys  = np.random.randint(0, W, alpha)
	
	sampled_response = response_maps[rand_xs, rand_ys, :]
	
	file_name = os.path.join('../data/tmp/',
							 'filter_reponse_' + str(i) + '.npy')

	np.save(file_name, sampled_response)

	return file_name

def compute_dictionary(num_workers=2):
	'''
	Creates the dictionary of visual words by clustering using k-means.

	[input]
	* num_workers: number of workers to process in parallel
	
	[saved]
	* dictionary: numpy.ndarray of shape (K,3F)
	'''
	
	# ----- TODO -----
	

	train_data = np.load("../data/train_data.npz")
	directory = '../data/tmp/'
	if not os.path.exists(directory):
		os.makedirs(directory)
		print('creating tmp directory ..')
	
	labels, image_paths = train_data['labels'], train_data['image_names']

	alpha = 270
	K = 150
	F = 20
	
	random_sampling_time = time.time()
	
	pool = multiprocessing.Pool(processes=num_workers)

	args = [[i, alpha, os.path.join('../data', image_path[0]), time.time()] 
			 	  for i, image_path in enumerate(image_paths)]
	f_list = pool.map(compute_dictionary_one_image, args)


	T    = len(f_list)
	filter_responses = np.zeros((T*alpha, 3*F))

	i = 0
	
	print('Random sampling took ', str(round(time.time() - random_sampling_time)), 'secs')
	print('PERFORMING KMEANS CLUSTERING ...')
	clustering_time = time.time()
	for file_name in f_list:
	
		f_data = np.load(file_name)
		filter_responses[i:i + alpha, :] = f_data
		i += alpha

	#use all cpus but one
	kmeans = sklearn.cluster.KMeans(n_clusters=K, n_jobs=-2).fit(filter_responses)

	print('Clustering took ', str(round(time.time() - clustering_time)), 'secs')
	
	dictionary = kmeans.cluster_centers_
	np.save('dictionary_alpha_'+str(alpha)+'_K_'+str(K)+'.npy', 
			dictionary)
		

