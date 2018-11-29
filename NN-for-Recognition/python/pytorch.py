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
    
def validate(valid_x, valid_y, batch_size, model):

	model.eval()	
	batches = get_random_batches(valid_x, valid_y, batch_size)
	total_loss = 0
	total_acc = 0

	for xb,yb in batches:
		#xb = np.array([xb[i, :].reshape(32, 32) for i in range(xb.shape[0])])
		X, y = to_variable(xb), to_variable(yb)		
		out  = model(X)
		
		pred = F.log_softmax(out, dim=1).data.max(1, keepdim=True)[1].int()
		predicted = pred.eq(y.data.view_as(pred).int())

		total_acc += predicted.sum()
		
		loss =  loss_fn(out, y.long())
		total_loss += loss.data.sum()
	
	return total_loss, total_acc/valid_x.shape[0]
	    
train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']

N = train_x.shape[0]
max_iters = 200
batch_size = 32
learning_rate = 2e-3
L2 = 1e-3
momentum = 0.9
train_y = np.argmax(train_y, axis=1)
valid_y = np.argmax(valid_y, axis=1)

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
optimizer = torch.optim.SGD(fc_model.parameters(), 
							lr=learning_rate)#,
							#momentum=momentum,
							#nesterov=True,
							#weight_decay=L2)
#optimizer = torch.optim.Adam(fc_model.parameters())

loss_fn = nn.CrossEntropyLoss() #


for itr in range(max_iters):
	total_loss = 0
	total_acc = 0
	#fc_model.train()
	cnn_model.train()
	for idx, (xb,yb) in enumerate(batches):
		optimizer.zero_grad()
		
		#xb = np.array([xb[i, :].reshape(32, 32) for i in range(xb.shape[0])])
		X, y = to_variable(xb), to_variable(yb)		
		out  = fc_model(X)
		#out  = cnn_model(X.unsqueeze(1))
		
		pred = F.log_softmax(out, dim=1).data.max(1, keepdim=True)[1].int()
		predicted = pred.eq(y.data.view_as(pred).int())

		total_acc += predicted.sum()
		
		loss =  loss_fn(out, y.long())
		loss.backward()
		optimizer.step()
		total_loss += loss.data.sum()

		#print("itr: {:02d}".format(idx*len(batches)), N)
	if itr % 2 == 0:
		val_loss, val_acc = validate(valid_x, valid_y, batch_size, fc_model)
		print("itr: {:02d} \t loss: {:.2f} \t acc: {:.2f} \t val acc: {:.2f}".format(itr,total_loss, total_acc/N, val_acc))
	

	
