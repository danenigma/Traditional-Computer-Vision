import numpy as np
import threading
import queue
import imageio
import os,time
import math
import visual_words
import skimage.measure
from sklearn.metrics import confusion_matrix

def get_image_feature_worker(task_queue):
	
	while True:
	
		(i, label, file_path, dictionary, SPM_layer_num, dict_size) = task_queue.get()
		print('['+str(i)+'] processing ' + file_path + ' : ', label)
		feature = get_image_feature(file_path, dictionary,  SPM_layer_num, dict_size)
				
		file_name = os.path.join('../data/tmp/hist_feats/', 'hist_'+ str(i) + '.npy')
		np.save(file_name, [feature, label]) 
		task_queue.task_done()
		
def build_recognition_system(num_workers=2):
	'''
	Creates a trained recognition system by generating training features from all training images.

	[input]
	* num_workers: number of workers to process in parallel

	[saved]
	* features: numpy.ndarray of shape (N,M)
	* labels: numpy.ndarray of shape (N)
	* dictionary: numpy.ndarray of shape (K,3F)
	* SPM_layer_num: number of spatial pyramid layers
	'''



	train_data = np.load("../data/train_data.npz")
	#-----TODO CHANGE to dictionary
	dictionary = np.load("dictionary.npy")
	# ----- TODO ----- parallelize 
	labels, image_paths = train_data['labels'], train_data['image_names']
	
	SPM_layer_num  = 3
	dict_size, F_3 = dictionary.shape
	N = len(image_paths)
	M = int(dict_size*(4**SPM_layer_num-1)/3)

	features  = np.zeros((N, M))
	
	directory = '../data/tmp/hist_feats/'
	
	if not os.path.exists(directory):
		os.makedirs(directory)
		print('creating hist tmp directory ..')

	
	training_time = time.time()
	
	task_queue  = queue.Queue(maxsize=0)
	num_threads = 2*num_workers
	
	for i in range(num_threads):
		worker = threading.Thread(target=get_image_feature_worker,
								  args=(task_queue,))
		worker.setDaemon(True)
		worker.start()

	
	for i, image_path in enumerate(image_paths):
		file_path = os.path.join('../data', image_path[0])
		label = labels[i]
		arg = [i, label, file_path, dictionary,  SPM_layer_num, dict_size]
		task_queue.put(arg)
			
	task_queue.join()
	
	f_list = os.listdir(directory)
	
	for index, f_name in enumerate(f_list):
		
		f_path = os.path.join(directory, f_name)
		f_data, label = np.load(f_path)
		features[index, :] = f_data
		
		# make sure the threading doesn't mess up the labels
		labels[index] = label
		
	
	np.savez('trained_system.npz', dictionary=dictionary,
								        features=features,  
								        labels=labels, 
								        SPM_layer_num=SPM_layer_num)

	print('Training took '+ str(round(time.time()-training_time))+ ' secs') 

def evaluate_recognition_system(num_workers=2):

	'''
	Evaluates the recognition system for all test images and returns the confusion matrix.

	[input]
	* num_workers: number of workers to process in parallel

	[output]
	* conf: numpy.ndarray of shape (8,8)
	* accuracy: accuracy of the evaluated system
	'''


	test_data = np.load("../data/test_data.npz")
	trained_system = np.load("trained_system.npz")
	# ----- TODO -----
	test_labels, image_paths = test_data['labels'], test_data['image_names']
	features = trained_system['features']
	train_labels = trained_system['labels']
	dictionary = trained_system['dictionary']
	SPM_layer_num= trained_system['SPM_layer_num']
	
		
	dict_size, F_3 = dictionary.shape


		
	#-----TODO------
	N_test = len(test_labels)
	prediction = np.zeros(N_test)
	test_time  = time.time()
	
	for i, image_path in enumerate(image_paths):
		
		if i==N_test:break	

	
		file_path  = os.path.join('../data', image_path[0])
		test_hist  = get_image_feature(file_path, dictionary,  SPM_layer_num, dict_size)	
		similarity = distance_to_set(test_hist, features)
		train_index   = np.argmax(similarity)
		prediction[i] = train_labels[train_index]
		print('['+str(i)+'] ' + str(round(time.time()-test_time))+' secs')
	
	accuracy = 	100*sum(prediction == test_labels[:N_test])/N_test

	conf = confusion_matrix(test_labels[:N_test], prediction, range(8))
	#conf = conf/np.sum(conf, axis=1)
	
	np.savez('result_N_alpha_270_K_150.npz', dictionary=dictionary,
						        features=features,  
						        train_labels=train_labels, 
						        SPM_layer_num=SPM_layer_num,
						        test_labels = test_labels,
						        prediction = prediction,
						        conf = conf,
						        accuracy = accuracy)
	
	return conf, accuracy
	
	
		
	
def get_image_feature(file_path, dictionary, layer_num, K):
	
	'''
	Extracts the spatial pyramid matching feature.

	[input]
	* file_path: path of image file to read
	* dictionary: numpy.ndarray of shape (K,3F)
	* layer_num: number of spatial pyramid layers
	* K: number of clusters for the word maps

	[output]
	* feature: numpy.ndarray of shape (K)
	'''


	# ----- TODO -----
	image   = imageio.imread(file_path) 
	wordmap = visual_words.get_visual_words(image, dictionary)	

	feature = get_feature_from_wordmap_SPM(wordmap, layer_num, K)	
	return feature

def distance_to_set(word_hist,histograms):
	
	'''
	Compute similarity between a histogram of visual words with all training image histograms.

	[input]
	* word_hist: numpy.ndarray of shape (K)
	* histograms: numpy.ndarray of shape (N,K)

	[output]
	* sim: numpy.ndarray of shape (N)
	'''
	
	# ----- TODO -----
	
	N, K = histograms.shape
	word_hist_dup = np.array([word_hist, ]*N)#duplicate for fast computation
	diff_of_hists = word_hist_dup - histograms
	min_of_hists  = np.zeros((N, K))
	min_of_hists[diff_of_hists>0]  = histograms[diff_of_hists>0]
	min_of_hists[diff_of_hists<=0] = word_hist_dup[diff_of_hists<=0]
	
	hist_intesection = np.sum(min_of_hists, axis=1)
	
	return hist_intesection




def get_feature_from_wordmap(wordmap, dict_size):
	
	'''
	Compute histogram of visual words.

	[input]
	* wordmap: numpy.ndarray of shape (H,W)
	* dict_size: dictionary size K

	[output]
	* hist: numpy.ndarray of shape (K)
	'''
	
	# ----- TODO -----
	hist, _ = np.histogram(wordmap, bins=dict_size, range=(0, dict_size-1))
	
	return hist
	



def get_feature_from_wordmap_SPM(wordmap, layer_num, dict_size):

	'''
	Compute histogram of visual words using spatial pyramid matching.

	[input]
	* wordmap: numpy.ndarray of shape (H,W)
	* layer_num: number of spatial pyramid layers
	* dict_size: dictionary size K

	[output]
	* hist_all: numpy.ndarray of shape (K*(4^layer_num-1)/3)
	'''
	
	# ----- TODO -----
	L = layer_num-1
	M = int(dict_size*(4**(L + 1)-1)/3)

	H, W = wordmap.shape
	
	fine_bin_n = 2**L #per dimension
	
	w_bin_size   = float(W)/fine_bin_n
	h_bin_size   = float(H)/fine_bin_n
	
	
	pyramid = []
	#append last layer (top layer)
	pyramid.append(np.zeros((fine_bin_n, fine_bin_n, dict_size)))
	#compute histograms of last layer
	for i in range(fine_bin_n):
		for j in range(fine_bin_n):
	
			block = wordmap[math.floor(i*h_bin_size):math.floor((i+1)*h_bin_size), 
							math.floor(j*w_bin_size):math.floor((j+1)*w_bin_size)]							
			
			#compute histogram of the block
			
			block_hist = get_feature_from_wordmap(block, dict_size)
			
			pyramid[0][i, j, :] = block_hist
		
	
	for layer_n in range(1, layer_num):
		layer_n_hists = skimage.measure.block_reduce(pyramid[layer_n-1], #use layer above
												block_size=(2, 2, 1),#sum histograms of 4 blocks 
												func=np.sum) 
		pyramid.append(layer_n_hists)
	
	#reverse sort the layers
	pyramid.reverse()
	#layer_n starts at 0
	
	final_hist_mat = []
	
	layer_w = 0
	
	for l in range(layer_num):
	
		if l == 0 or l == 1:
	
			layer_w = math.pow(2, -L)
			
		else:
	
			layer_w = math.pow(2, l-L-1)
			
		
		final_hist_mat.append(layer_w*pyramid[l].reshape(-1, dict_size))

	features = np.concatenate(final_hist_mat, axis=0).reshape(1, -1).squeeze(0)


	return features/features.sum()
	






	

