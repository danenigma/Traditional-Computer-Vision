import numpy as np
import cv2
import cv2
import numpy as np
from scipy.ndimage import shift
from scipy.interpolate import RectBivariateSpline

img = np.random.randint(0, 255, (500, 500)).astype('uint8')

#num_rows, num_cols = img.shape[:2]
H, W = img.shape

x = np.linspace(0, H, H)
y = np.linspace(0, W, W)
print('X: ', len(x), 'Y: ', len(y))
It1_BiRect = RectBivariateSpline(x, y, img)
print(It1_BiRect.ev([[1, 2],[1,5]], [[1, 3],[2,6]]))
print('im: ', img[:6, :6])
#It1_BiRect.ev()
#translation_matrix = np.float32([ [1,0,200], [0,1,210] ])
#img_translation = shift(img, (100.5, 100.5, 0.0))
#cv2.imshow('Translation', img_translation )
#cv2.waitKey(10000)
