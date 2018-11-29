import numpy as np
import pickle
import string
import scipy.io
from nn import *
import matplotlib.pyplot as plt

valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

params = pickle.load(open('q5_weights.pickle','rb'))
valid_x = valid_data['valid_data']
print(valid_x.shape)

samples = [5, 55, 305, 355, 505, 555, 1005, 1055, 2805, 2855]


h1 = forward(valid_x[samples, :],params,'layer1',relu)
h2 = forward(h1,params,'layer2',relu)
h3 = forward(h2,params,'layer3',relu)
out = forward(h3,params,'output',sigmoid)

for i, s in enumerate(samples):
    plt.subplot(2,1,1)
    plt.imshow(valid_x[s].reshape(32,32).T)
    plt.subplot(2,1,2)
    plt.imshow(out[i].reshape(32,32).T)
    plt.show()

