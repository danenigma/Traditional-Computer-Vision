import numpy as np
import scipy.ndimage
from skimage.measure import block_reduce
import util
import os,time
#for testing 
import torch.nn as nn
import torch
import deep_recog
def extract_deep_feature(x,vgg16_weights):
	'''
	Extracts deep features from the given VGG-16 weights.

	[input]
	* x: numpy.ndarray of shape (H,W,3)
	* vgg16_weights: numpy.ndarray of shape (L,3)

	[output]
	* feat: numpy.ndarray of shape (K)
	'''
	#-------TODO---------

	for index, vgg_weights in enumerate(vgg16_weights[:-1]):
		
		#print(index, vgg_weights[0])
	
		
		if vgg_weights[0] == "conv2d":
			x = multichannel_conv2d(x, vgg_weights[1], vgg_weights[2])

		if vgg_weights[0] == "relu":
			x = relu(x)

		if vgg_weights[0] == "maxpool2d":
			x = max_pool2d(x, vgg_weights[1])

		if vgg_weights[0] == "linear":

			x = linear(x, vgg_weights[1], vgg_weights[2])

		if index == 30:#after conv flatten 
			x = x.flatten()

		#print('Done computing layer: [' + str(index) + '] ' + vgg_weights[0])	
	
	return x
	
def multichannel_conv2d(x,weight,bias):
	'''
	Performs multi-channel 2D convolution.

	[input]
	* x: numpy.ndarray of shape (H, W, input_dim)
	* weight: numpy.ndarray of shape (output_dim,input_dim,kernel_size,kernel_size)
	* bias: numpy.ndarray of shape (output_dim)

	[output]
	* feat: numpy.ndarray of shape (H,W,output_dim)
	'''
	
	(H,W,input_dim) = x.shape
	(output_dim,input_dim,kernel_size,kernel_size) = weight.shape
	
	feat = np.zeros((H,W,output_dim))

	for i in range(output_dim):#for each filter
		for ch in range(input_dim):#for each channel
		
			w = np.flipud(np.fliplr(weight[i, ch, :, :]))
			
			feat[:,:,i] += scipy.ndimage.convolve(x[:, :, ch],
						                          w, 
						                          mode='constant', 
						                          cval=0.0)
			
			"""
			#using correlation
			feat[:,:,i] += scipy.ndimage.correlate(x[:, :, ch],
						                           weight[i, ch, :, :], 
						                           mode='constant', 
						                           cval=0.0) 
			"""
					                          
		feat[:,:,i] += bias[i]
		
	return feat

def relu(x):
	'''
	Rectified linear unit.

	[input]
	* x: numpy.ndarray

	[output]
	* y: numpy.ndarray
	'''
	return np.maximum(0, x)
	
	
def max_pool2d(x,size):
	'''
	2D max pooling operation.

	[input]
	* x: numpy.ndarray of shape (H,W,input_dim)
	* size: pooling receptive field

	[output]
	* y: numpy.ndarray of shape (H/size,W/size,input_dim)
	'''
	out = block_reduce(x, block_size=(size, size, 1), func=np.max)
	return out

def linear(x,W,b):
	'''
	Fully-connected layer.

	[input]
	* x: numpy.ndarray of shape (input_dim)
	* weight: numpy.ndarray of shape (output_dim,input_dim)
	* bias: numpy.ndarray of shape (output_dim)

	[output]
	* y: numpy.ndarray of shape (output_dim)
	'''
	out = np.matmul(x, W.T) + b
	return out



if __name__=='__main__':
	
	image_np = np.random.randn(224, 224, 3)
	
	"""
	image = deep_recog.preprocess_image(image_np)
	
	conv = nn.Conv2d(in_channels=3, out_channels= 10,
		                        kernel_size =(5,5), padding= (2,2)).double()
	relu_torch = nn.ReLU()
	max_torch  = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False)
	linear_torch = nn.Linear(in_features=25088, out_features=4096, bias=True)
	
	wl, bl = linear_torch.weight.data.numpy(), linear_torch.bias.data.numpy()
	
	print('wl shape: ', wl.shape)
	w, b = conv.weight.data.numpy(), conv.bias.data.numpy()


	out    = conv(torch.autograd.Variable(image))
	my_out = multichannel_conv2d(image_np, w, b)
	my_out_tensor = deep_recog.preprocess_image(my_out)
	print('Conv: ', np.allclose(out.data.numpy(), my_out_tensor.numpy()))
	
	my_out = relu(my_out)
	out = relu_torch(out)
	my_out_tensor = deep_recog.preprocess_image(my_out)	
	print('Relu: ', np.allclose(out.data.numpy(), my_out_tensor.numpy()))
	
	my_out =  max_pool2d(my_out, 2)
	out = max_torch(out)
	out_np = out.permute(0,2,3,1).squeeze(0)
	print('Max_pool2d: ', np.allclose(out_np.data.numpy(), my_out))
	
	x = np.random.randn(25088).astype('float')
	x_torch = torch.from_numpy(x).float()
	
	my_lin_out    = linear(x,wl,bl)
	torch_lin_out = linear_torch(torch.autograd.Variable(x_torch))
	print(np.min(my_lin_out), np.max(my_lin_out), np.mean(my_lin_out), np.std(my_lin_out))
	print('Linear: ', np.allclose(my_lin_out, torch_lin_out.data.numpy(), atol=2e-06))
	
	
	
	vgg16_weights = util.get_VGG16_weights()

	extract_deep_feature(image_np, vgg16_weights)
	"""

	
