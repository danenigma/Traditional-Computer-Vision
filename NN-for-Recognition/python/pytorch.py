import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from skimage.measure import compare_psnr as psnr

class Flatten(nn.Module):
	"""
	Implement a simple custom module that reshapes (n, m, 1, 1) tensors to (n, m).
	"""
	def __init__(self):
		super(Flatten, self).__init__()

	def forward(self,x):
		n = x.size()[0]
		return x.view(n, -1)

def to_variable(tensor):
	# Tensor -> Variable (on GPU if possible)
	tensor = torch.from_numpy(tensor).type(torch.FloatTensor)
	return torch.autograd.Variable(tensor)

def get_random_batches(x,y,batch_size):
    batches = []
    N = x.shape[0] 
    n_batches = N // batch_size
    
    for i in range(n_batches):
    	randIdx = np.random.choice(N, batch_size, replace=False)
    	batch = (x[randIdx, :], y[randIdx])
    	batches.append(batch)
    return batches
    
    
train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']

N = train_x.shape[0]
max_iters = 100
batch_size = 128
learning_rate = 1e-3
L2 = 1e-3
momentum = 0.9
train_y = np.argmax(train_y, axis=1)
batches = get_random_batches(train_x, train_y, batch_size)

fc_model = nn.Sequential(
		   nn.Linear(1024, 64),
		   nn.Linear(64, 36))
cnn_model = nn.Sequential(
			nn.Conv2d(1, 64, 3),
			nn.ReLU(),
			nn.Conv2d(64, 64, 3),
			nn.ReLU(),
			nn.Conv2d(64, 64, 3),
			nn.ReLU(),
			nn.AvgPool2d(6),
			Flatten(),
		  	nn.Linear(1024, 64),
		   	nn.Linear(64, 36))

print(fc_model)   
optimizer = torch.optim.SGD(cnn_model.parameters(), 
							lr=learning_rate,
							momentum=momentum,
							nesterov=True,
							weight_decay=L2)
loss_fn = nn.NLLLoss() #


for itr in range(max_iters):
	total_loss = 0
	total_acc = 0
	fc_model.train()

	for xb,yb in batches:
		xb = np.array([xb[i, :].reshape(32, 32) for i in range(xb.shape[0])])
		
		optimizer.zero_grad()
		X, y = to_variable(xb), to_variable(yb)
		
		
		out  = F.log_softmax(cnn_model(X.unsqueeze(1)))
		pred = out.data.max(1, keepdim=True)[1].int()
		predicted = pred.eq(y.data.view_as(pred).int())

		total_acc += predicted.sum()
		loss = loss_fn(out, y.long())
		loss.backward()
		optimizer.step()
		total_loss += loss.data.sum()
	if itr % 2 == 0:
		print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss,total_acc/N))
	

