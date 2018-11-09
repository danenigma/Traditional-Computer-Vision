'''
Q4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter3
'''
import numpy as np
import submission as sub
import helper

corresp = np.load('../data/some_corresp.npz')
intrinsics = np.load('../data/intrinsics.npz')

