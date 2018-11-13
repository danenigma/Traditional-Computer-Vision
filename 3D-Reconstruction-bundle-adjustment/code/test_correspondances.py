import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import submission as sub
import helper


data = np.load('../data/some_corresp.npz')
im1 = plt.imread('../data/im1.png')
im2 = plt.imread('../data/im2.png')

N = data['pts1'].shape[0]
M = 640

F8 = sub.eightpoint(data['pts1'], data['pts2'], M)
print('F8: ', F8, )

