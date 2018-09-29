from alignChannels import alignChannels
import scipy.misc
import numpy as np

# Problem 1: Image Alignment

# 1. Load images (all 3 channels)
red = np.load('../data/red.npy') #None
green = np.load('../data/green.npy')
blue = np.load('../data/blue.npy')


# 2. Find best alignment
rgbResult = alignChannels(red, green, blue)
# 3. save result to rgb_output.jpg (IN THE "results" FOLDER)
scipy.misc.imsave('results/rgb_output.jpg', rgbResult)

