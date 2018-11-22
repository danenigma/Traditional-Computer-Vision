import numpy as np
import scipy.io
from nn import *

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']

max_iters = 50
# pick a batch size, learning rate
batch_size = 256
learning_rate = 1e-3
hidden_size = 64

batches = get_random_batches(train_x,train_y,batch_size)
batch_num = len(batches)

params = {}


# initialize layers here

initialize_weights(1024, 64, params, 'layer1')
initialize_weights(64, 36, params, 'output')


# with default settings, you should get loss < 150 and accuracy > 80%
for itr in range(max_iters):
	total_loss = 0
	total_acc = 0
	for xb,yb in batches:
		# training loop can be exactly the same as q2!
		h1 = forward(xb, params,'layer1')
		probs = forward(h1, params, 'output', softmax)
		loss, acc = compute_loss_and_acc(yb, probs)

		total_loss += loss
		total_acc += acc

		delta1 = probs
		yb_idx = np.argmax(yb, axis=1)
		delta1[np.arange(probs.shape[0]), yb_idx] -= 1
		delta2 = backwards(delta1, params, 'output', linear_deriv)
		backwards(delta2, params, 'layer1', sigmoid_deriv)
	
		params['Wlayer1'] -= learning_rate*params['grad_Wlayer1']
		params['blayer1'] -= learning_rate*params['grad_blayer1']
		params['Woutput'] -= learning_rate*params['grad_Woutput']
		params['boutput'] -= learning_rate*params['grad_boutput']
	total_acc /= batch_num
	if itr % 2 == 0:
		print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss,total_acc))

# run on validation set and report accuracy! should be above 75%
h1 = forward(valid_x, params,'layer1')
probs = forward(h1, params, 'output', softmax)
_, valid_acc = compute_loss_and_acc(valid_y, probs)

#valid_acc = None

print('Validation accuracy: ', valid_acc)
if False: # view the data
    for crop in valid_x:
        import matplotlib.pyplot as plt
        plt.imshow(crop.reshape(32,32).T)
        plt.show()
        
import pickle
saved_params = {k:v for k,v in params.items() if '_' not in k}
with open('q3_weights.pickle', 'wb') as handle:
    pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Q3.1.3
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

fig = plt.figure(1, (16., 16.))
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(8, 8),  # creates 2x2 grid of axes
                 axes_pad=0.1,  # pad between axes in inch.
                 )

for i in range(64):
	im = params['Wlayer1'][:, i].reshape(32, 32)
	grid[i].imshow(im)  # The AxesGrid object work as a list of axes.

plt.show()

'''
# Q3.1.3
confusion_matrix = np.zeros((train_y.shape[1],train_y.shape[1]))

import string
plt.imshow(confusion_matrix,interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.show()
'''
