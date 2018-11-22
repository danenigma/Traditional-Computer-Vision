import numpy as np
# you should write your functions in nn.py
from nn import *
from util import *


# fake data
# feel free to plot it in 2D
# what do you think these 4 classes are?
g0 = np.random.multivariate_normal([3.6,40],[[0.05,0],[0,10]],10)
g1 = np.random.multivariate_normal([3.9,10],[[0.01,0],[0,5]],10)
g2 = np.random.multivariate_normal([3.4,30],[[0.25,0],[0,5]],10)
g3 = np.random.multivariate_normal([2.0,10],[[0.5,0],[0,10]],10)
x = np.vstack([g0,g1,g2,g3])
# we will do XW + B
# that implies that the data is N x D

# create labels
y_idx = np.array([0 for _ in range(10)] + [1 for _ in range(10)] + [2 for _ in range(10)] + [3 for _ in range(10)])
# turn to one_hot
y = np.zeros((y_idx.shape[0],y_idx.max()+1))
y[np.arange(y_idx.shape[0]),y_idx] = 1

params = {}
initialize_weights(2, 4, params,'layer1')
probs = forward(x, params, 'layer1', softmax)
delta1 = probs
delta1[np.arange(probs.shape[0]),y_idx] -= 1
backwards(delta1, params, name='layer1', activation_deriv=linear_deriv)

print('delta: ', delta1)
print('probs: ', probs==delta1)

import copy
params_orig = copy.deepcopy(params)

eps = 1e-6

for k,v in params.items():
	if '_' in k: 
		continue
	
	if len(params[k].shape) == 1:
		for i in range(params[k].shape[0]): 
			params_temp = copy.deepcopy(params)
			params[k][i] += eps
			probs     = forward(x, params,'layer1')
			loss_p, _ = compute_loss_and_acc(y, probs)
			params = copy.deepcopy(params_temp)
			params[k][i] -=eps

			probs = forward(x, params,'layer1')
			loss_m, _ = compute_loss_and_acc(y, probs)
			params = copy.deepcopy(params_temp)
			params['grad_' + k][i] = (loss_p - loss_m)/(2*eps)

	
	else:
		for i in range(params[k].shape[0]):
			for j in range(params[k].shape[1]):
			 
				params[k][i, j] += eps
				probs = forward(x, params,'layer1')
				loss_p, _ = compute_loss_and_acc(y, probs)
				params[k][i, j] -=2*eps
				probs = forward(x, params,'layer1')
				loss_m, _ = compute_loss_and_acc(y, probs)
				params['grad_' + k][i, j] = (loss_p - loss_m)/(2*eps)
				params[k][i] +=eps

total_error = 0
for k in params.keys():
    if 'grad_' in k:
        # relative error
        err = np.abs(params[k] - params_orig[k])/np.maximum(np.abs(params[k]),np.abs(params_orig[k]))
        err = err.sum()
        print('{} {:.2e}'.format(k, err))
        total_error += err
# should be less than 1e-4
print('total {:.2e}'.format(total_error))
print('grad_W', params['grad_Wlayer1'])
print('numeric grad_W', params_orig['grad_Wlayer1'])
print('grad_b', params['grad_blayer1'])
print('numeric grad_b', params_orig['grad_blayer1'])
