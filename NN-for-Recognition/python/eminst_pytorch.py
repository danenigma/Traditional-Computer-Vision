import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision import transforms

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
	return torch.autograd.Variable(tensor)


emnist_train = torchvision.datasets.EMNIST('../data/', 
										  'balanced',
										  train = True, 
										  download = False,
										  transform =transforms.ToTensor())
										  
emnist_test = torchvision.datasets.EMNIST('../data/', 
										  'balanced',
										  train = False, 
										  download = False,
										  transform = transforms.ToTensor())
										  
train_data_loader = torch.utils.data.DataLoader(emnist_train,
                                          batch_size = 4,
                                          shuffle=True)
                                          
cnn_model = nn.Sequential(
			nn.Conv2d(1, 64, 3),
			nn.ReLU(),
			nn.Conv2d(64, 64, 3),
			nn.ReLU(),
			nn.Conv2d(64, 64, 3),
			nn.ReLU(),
			nn.AvgPool2d(6),
			Flatten(),
		  	nn.Linear(576, 64),
		   	nn.Linear(64, 47)
		   	)

print(cnn_model)   
learning_rate = 1e-3
max_iters = 1
optimizer = torch.optim.SGD(cnn_model.parameters(), 
							lr=learning_rate)
							#momentum=momentum,
							#nesterov=True,
							#weight_decay=L2)
#optimizer = torch.optim.Adam(fc_model.parameters())

loss_fn = nn.CrossEntropyLoss() #
                                          
                                          
for itr in range(max_iters):
	total_loss = 0
	total_acc = 0
	cnn_model.train()

	for idx, (xb,yb) in enumerate(train_data_loader):
		optimizer.zero_grad()
		
		X, y = to_variable(xb), to_variable(yb)	
		out  = cnn_model(X)
		pred = F.log_softmax(out, dim=1).data.max(1, keepdim=True)[1].int()
		predicted = pred.eq(y.data.view_as(pred).int())

		total_acc += predicted.sum()
		
		loss =  loss_fn(out, y.long())
		loss.backward()
		optimizer.step()

		total_loss += loss.data.sum()
		
		#print("itr: {:02d}".format(idx*len(batches)), N)
	if itr % 2 == 0:
		print("itr: {:02d} \t loss: {:.2f} \t acc: {:.2f} \t val acc: {:.2f}".format(itr,total_loss, total_acc/N))

