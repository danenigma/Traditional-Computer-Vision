import numpy as np
import multiprocessing
import threading
import queue
import imageio
import os,time
import torch
import skimage.transform
import torchvision.transforms
import util
import network_layers
import scipy.spatial.distance
from sklearn.metrics import confusion_matrix

def build_recognition_system(vgg16, num_workers=2):
	'''
	Creates a trained recognition system by generating training features from all training images.

	[input]
	* vgg16: prebuilt VGG-16 network.
	* num_workers: number of workers to process in parallel

	[saved]
	* features: numpy.ndarray of shape (N,K)
	* labels: numpy.ndarray of shape (N)
	'''


	train_data = np.load("../data/train_data.npz")

	# ----- TODO -----
	
	training_time = time.time()
	
	directory = '../data/tmp/vgg_feats/'
	if not os.path.exists(directory):
		os.makedirs(directory)
		print('creating tmp directory ..')

	labels, image_paths = train_data['labels'], train_data['image_names']
	pool = multiprocessing.Pool(processes=num_workers)

	args = [[i, os.path.join('../data', image_path[0]), vgg16, time.time()] 
			 	  for i, image_path in enumerate(image_paths)]

	result_list = pool.map(get_image_feature, args)
	

	T    = len(result_list)
	
	deep_feat  = np.zeros((T, 4096))
	new_labels = np.zeros(T)
	print('Computing deep dictionary ...')

	for result in enumerate(result_list):
		
		f_name, index = result[1]
		f_data = np.load(f_name)
		deep_feat[index, :] = f_data
		new_labels[index] =  labels[index]
		
	np.savez('trained_system_deep.npz',
			  deep_feat=deep_feat, 
			  labels=new_labels)

	print('Done!!')
	print('Training took '+ str(round(time.time()-training_time))+ ' secs') 

def evaluate_recognition_system(vgg16,num_workers=2):
	'''
	Evaluates the recognition system for all test images and returns the confusion matrix.

	[input]
	* vgg16: prebuilt VGG-16 network.
	* num_workers: number of workers to process in parallel

	[output]
	* conf: numpy.ndarray of shape (8,8)
	* accuracy: accuracy of the evaluated system
	'''

	
	test_data = np.load("../data/test_data.npz")
	trained_system = np.load("trained_system_deep.npz")
	# ----- TODO -----
	test_labels, image_paths = test_data['labels'], test_data['image_names']
	deep_feat     = trained_system['deep_feat']
	train_labels  = trained_system['labels']
	
	
		
	#-----TODO------
	N_test = len(test_labels)
	prediction = np.zeros(N_test)
	
	for i, image_path in enumerate(image_paths):

		file_path  = os.path.join('../data', image_path[0])
		
		image = imageio.imread(file_path)
		
		if len(image.shape)==2:
			image = np.stack((image,)*3, -1)
		elif(image.shape[2]==4):
			print('I Found one')
			image = image[:, :, :3]	
		image = preprocess_image(image)

		image_tensor = torch.autograd.Variable(image)

		#vgg_conv  = torch.nn.Sequential(*list(vgg16.features.children()))
		vgg_fc7   = torch.nn.Sequential(*list(vgg16.classifier.children())[:-2])

		conv_feat = vgg16.features(image_tensor)		
		fc7_feat  = vgg_fc7(conv_feat.view(-1)).unsqueeze(0).data.numpy()
			
		
		print('image feat extracted [', i, ']')
		similarity = distance_to_set(fc7_feat, deep_feat)
		train_index   = np.argmax(similarity)
		prediction[i] = train_labels[train_index]
		
	accuracy = 	100*sum(prediction == test_labels[:N_test])/N_test

	conf = confusion_matrix(test_labels[:N_test], prediction, range(8))
	conf = conf/np.sum(conf, axis=1)
	np.savez('result_deep_with_normalization.npz', features = deep_feat,  
						        train_labels = train_labels, 
						        test_labels = test_labels,
						        prediction = prediction,
						        conf = conf,
						        accuracy = accuracy)
	
	
	return conf, accuracy


def preprocess_image(image):
	'''
	Preprocesses the image to load into the prebuilt network.

	[input]
	* image: numpy.ndarray of shape (H,W,3)

	[output]
	* image_processed: torch.Tensor of shape (3,H,W)
	'''
	
	# ----- TODO -----
	image = skimage.transform.resize(image, (224, 224))
	trans = torchvision.transforms.Compose([
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])
	
	return trans(np.array(image)).unsqueeze(0)

def get_image_feature(args):
	'''
	Extracts deep features from the prebuilt VGG-16 network.
	This is a function run by a subprocess.
 	[input]
	* i: index of training image
	* image_path: path of image file
	* vgg16: prebuilt VGG-16 network.
	* time_start: time stamp of start time
 	[saved]
	* feat: evaluated deep feature
	'''

	i, image_path, vgg16, time_start = args

	# ----- TODO -----
	
	image = imageio.imread(image_path)

	if len(image.shape)==2:
		image = np.stack((image,)*3, -1)
	elif(image.shape[2]==4):
		image = image[:, :, :3]	

	image = preprocess_image(image)
	
	image_tensor = torch.autograd.Variable(image)

	#vgg_conv  = torch.nn.Sequential(*list(vgg16.features.children()))
	vgg_fc7   = torch.nn.Sequential(*list(vgg16.classifier.children())[:-2])
	conv_feat = vgg16.features(image_tensor)		
	fc7_feat  = vgg_fc7(conv_feat.view(-1))
	#fc7_feat   = vgg16(image_tensor)
		
	file_name = os.path.join('../data/tmp/vgg_feats/','vgg_feats_'+str(i)+'.npy')
	np.save(file_name, fc7_feat.data.numpy())
		    
	print('Done Processing image [' + str(i) + ']')
	
	return [file_name, i]

def distance_to_set(feature,train_features):
	'''
	Compute distance between a deep feature with all training image deep features.

	[input]
	* feature: numpy.ndarray of shape (K)
	* train_features: numpy.ndarray of shape (N,K)

	[output]
	* dist: numpy.ndarray of shape (N)
	'''

	# ----- TODO -----
	dist = -1*scipy.spatial.distance.cdist(feature, train_features)
	
	return dist

