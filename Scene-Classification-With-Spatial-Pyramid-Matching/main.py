import numpy as np
import torchvision
import util
import matplotlib.pyplot as plt
import visual_words
import visual_recog
import deep_recog
import skimage.io
# my headers
import time
import torch
if __name__ == '__main__':

	num_cores = util.get_num_CPU()

	#path_img = "../data/kitchen/sun_aasmevtpkslccptd.jpg"
	path_img = "../data/kitchen/word_map1_in.jpg"
	image = skimage.io.imread(path_img)
	
	filter_responses = visual_words.extract_filter_responses(image)
	util.display_filter_responses(filter_responses)
	#args = (0, 100, path_img, time.time())
	#visual_words.compute_dictionary_one_image(args)	
	#visual_words.compute_dictionary(num_workers=num_cores)
	dictionary = np.load('dictionary.npy')
	K, F_3     = dictionary.shape
	wordmap    = visual_words.get_visual_words(image, dictionary)
	
	filename = 'wordmap_1.jpg'
	util.save_wordmap(wordmap, filename)
	#visual_recog.get_feature_from_wordmap(wordmap, K))			

	layer_num = 3

	SPM_feat = visual_recog.get_feature_from_wordmap_SPM(wordmap, layer_num, K)
	print(SPM_feat.shape, SPM_feat.sum(), image.shape[0]*image.shape[1])
	
	#SPM_feat = visual_recog.get_image_feature(path_img, dictionary, layer_num, K)
	#K = SPM_feat.shape[0]
	#N = 1500
	#word_hist  = np.random.randn(K)
	#histograms = np.random.randn(N, K)
	#histograms[50, :] = SPM_feat
	#histograms[51, :] = np.ones(K)/np.ones(K).sum()
	
	
	#print(histograms.shape)
	#distance = visual_recog.distance_to_set(SPM_feat, histograms)
	#print(distance.shape, distance[50], distance[51])
	#visual_recog.build_recognition_system(num_workers=num_cores)
	#conf, accuracy  = visual_recog.evaluate_recognition_system(num_workers=num_cores)
	
	#print(conf, accuracy)
	#print(np.diag(conf).sum()/conf.sum())

	#vgg16 = torchvision.models.vgg16(pretrained=True).double()
	#modules = list(resnet.children())[:-1]      # delete the last fc layer.
	#self.resnet  = nn.Sequential(*modules)
	#print(vgg16)
	#vgg16.eval()

	#deep_recog.build_recognition_system(vgg16,num_workers=num_cores//2)
	#print('Training Done.')
	#conf, acc = deep_recog.evaluate_recognition_system(vgg16, num_workers=num_cores//2)
	#print(conf, acc)
	#print(np.diag(conf).sum()/conf.sum())
	#wordmap = np.random.randn(800, 400)
	
	
