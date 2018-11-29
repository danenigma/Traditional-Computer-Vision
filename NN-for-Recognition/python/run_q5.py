import numpy as np
import scipy.io
from nn import *
from collections import Counter

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

# we don't need labels now!
train_x = train_data['train_data']
valid_x = valid_data['valid_data']

max_iters = 100
# pick a batch size, learning rate
batch_size = 36 
learning_rate =  3e-5
hidden_size = 32
lr_rate = 20
batches = get_random_batches(train_x,np.ones((train_x.shape[0],1)),batch_size)
batch_num = len(batches)

params = Counter()

# initialize layers here
initialize_weights(1024, 32, params, 'layer1')
initialize_weights(32, 32, params, 'layer2')
initialize_weights(32, 32, params, 'layer3')
initialize_weights(32, 1024, params, 'output')

training_loss = []
# should look like your previous training loops
for itr in range(max_iters):
	total_loss = 0
	for xb,_ in batches:

		h = forward(xb, params,'layer1', relu)
		h = forward(h, params,'layer2', relu)
		h = forward(h, params,'layer3', relu)
		output = forward(h, params,'output',  sigmoid)
		
		total_loss  += np.sum(np.square((output - xb)))
	
		delta = 2.0*(output - xb)

		delta = backwards(delta, params, 'output', sigmoid_deriv)
		delta = backwards(delta, params, 'layer3', relu_deriv)
		delta = backwards(delta, params, 'layer2', relu_deriv)
		backwards(delta, params, 'layer1', relu_deriv)

		

		params['mw_layer1'] =0.9*params['mw_layer1'] - learning_rate*params['grad_Wlayer1']
		params['mb_layer1'] =0.9*params['mb_layer1'] - learning_rate*params['grad_blayer1']
		params['mw_layer2'] =0.9*params['mw_layer2'] - learning_rate*params['grad_Wlayer2']
		params['mb_layer2'] =0.9*params['mb_layer2'] - learning_rate*params['grad_blayer2']
		params['mw_layer3'] =0.9*params['mw_layer3'] - learning_rate*params['grad_Wlayer3']
		params['mb_layer3'] =0.9*params['mb_layer3'] - learning_rate*params['grad_blayer3']
		params['mw_output'] =0.9*params['mw_output'] - learning_rate*params['grad_Woutput']
		params['mb_output'] =0.9*params['mb_output'] - learning_rate*params['grad_boutput']
	
		params['Wlayer1'] += params['mw_layer1']
		params['blayer1'] += params['mb_layer1']
		params['Wlayer2'] += params['mw_layer2']
		params['blayer2'] += params['mb_layer2']
		params['Wlayer3'] += params['mw_layer3']
		params['blayer3'] += params['mb_layer3']
		params['Woutput'] += params['mw_output']
		params['boutput'] += params['mb_output']
		# training loop can be exactly the same as q2!
		# your loss is now squared error
		# delta is the d/dx of (x-y)^2
		# to implement momentum
		#   just use 'm_'+name variables
		#   to keep a saved value over timestamps
		#   params is a Counter(), which returns a 0 if an element is missing
		#   so you should be able to write your loop without any special conditions
	training_loss.append(total_loss)	
	if itr % 2 == 0:
		print("itr: {:02d} \t loss: {:.2f}".format(itr,total_loss))
	if itr % lr_rate == lr_rate-1:
		learning_rate *= 0.9
# visualize some results
# Q5.3.1
import pickle
saved_params = {k:v for k,v in params.items() if '_' not in k}
with open('q5_weights.pickle', 'wb') as handle:
    pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

import matplotlib.pyplot as plt

fig1, ax = plt.subplots()
ax.plot(np.arange(max_iters), np.array(training_loss))
plt.xlabel('Epochs')
plt.ylabel('Total Loss')
plt.show()

h1 = forward(xb,params,'layer1',relu)
h2 = forward(h1,params,'layer2',relu)
h3 = forward(h2,params,'layer3',relu)
out = forward(h3,params,'output',sigmoid)
for i in range(5):
    plt.subplot(2,1,1)
    plt.imshow(xb[i].reshape(32,32).T)
    plt.subplot(2,1,2)
    plt.imshow(out[i].reshape(32,32).T)
    plt.show()


from skimage.measure import compare_psnr as psnr
# evaluate PSNR
# Q5.3.2
h1 = forward(valid_x,params,'layer1',relu)
h2 = forward(h1,params,'layer2',relu)
h3 = forward(h2,params,'layer3',relu)
recon_valid = forward(h3,params,'output',sigmoid)
total = []
for pred,gt in zip(recon_valid,valid_x):
    total.append(psnr(gt,pred))
print('psnr: ', np.array(total).mean())

