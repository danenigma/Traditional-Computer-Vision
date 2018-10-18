import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanade_new(It0, It1, rect, p0 = np.zeros(2)):

	x1, y1, x2, y2  = rect[0], rect[1], rect[2], rect[3]

	p, threshold = p0, 0.1


	H, W = It1.shape
	x = np.linspace(0, H, H)
	y = np.linspace(0, W, W)

	It0_BiRect = RectBivariateSpline(x, y, It0)	
	It1_BiRect = RectBivariateSpline(x, y, It1)	
	[X, Y] = np.meshgrid(np.linspace(x1, np.ceil(x2), np.ceil(x2)-x1), 
						 np.linspace(y1, np.ceil(y2), np.ceil(y2)-y1))
						 	
	template = It0_BiRect.ev(X, Y);
	[Fx, Fy] = np.gradient(template);
	A = np.stack((Fx.flatten(),Fy.flatten()), axis=1)

	H = np.matmul(A.T, A)
	print(H)
	while True:

		[X1, Y1] = np.meshgrid(np.linspace(x1+p[0], np.ceil(x2+p[0]), x2-x1), 
							   np.linspace(y1+p[1], np.ceil(y2+p[1]), y2-y1))

		errorImg = It1_BiRect.ev(X1, Y1) - template

		delta = np.linalg.lstsq(H, np.matmul(A.T, errorImg.flatten()))[0]
		
		p = p - delta
		print(np.linalg.norm(delta))
		if np.linalg.norm(delta) < threshold:
			break

	return p
