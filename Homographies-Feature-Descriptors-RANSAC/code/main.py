import cv2
import os
import numpy as np
import BRIEF
import keypointDetect
import matplotlib.pyplot as plt

levels = [-1,0,1,2,3,4]
#im = cv2.imread('../data/model_chickenbroth.jpg')
#im = cv2.imread('../data/prince_book.jpeg')
locsDoG, gaussian_pyramid = keypointDetect.DoGdetector(im)

N, _ = locsDoG.shape

print('N:', N)
im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
plt.imshow(im, cmap='gray')
for i in range(N):
	plt.plot([locsDoG[i][0]],[locsDoG[i][1]],'ro')
plt.show()
"""
for i in range(N):
	print(locsDoG[i])
	cv2.circle(im, (locsDoG[i][0], locsDoG[i][1]), 1, (0,0,255), 1)
cv2.imshow('out', im)
cv2.waitKey(0)
"""
#loc, desc = BRIEF.briefLite(im)
#print(loc.shape, desc.shape)
	
print('Done!!')

