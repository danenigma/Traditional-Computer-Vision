import numpy as np
from scipy.interpolate import RectBivariateSpline
def LK_correct(It0, It1, rect0, p0):

	x1 = rect0[0];
	x2 = rect0[2];
	y1 = rect0[1];
	y2 = rect0[3];
	
	x_range = np.linspace(y1, y2, y2-y1)
	y_range = np.linspace(x1, x2, x2-x1)
	
	[xt,yt] = np.meshgrid(x_range, y_range)
	
	H, W = It1.shape
	x = np.linspace(0, H, H)
	y = np.linspace(0, W, W)

	It_BiRect    = RectBivariateSpline(x, y, It0)	
	It1_BiRect   = RectBivariateSpline(x, y, It1)	

	It  = It_BiRect.ev(xt, yt)
	
	[Itx,Ity] = np.gradient(It)
	
	A = np.stack((Itx.flatten(),Ity.flatten()), axis=1)	
	H = np.matmul(A.T, A);


	p = p0
	tol = 0.001

	while True:
		
		warpedImg = WarpImg(It1_BiRect, p, rect0)
		
		b = It - warpedImg;
		b = np.matmul(A.T, b.flatten())
		del_p = np.matmul(np.linalg.inv(H), b)
		p = p + del_p
		print(np.linalg.norm(del_p))
		if np.linalg.norm(del_p)<tol:
			break
	return p

def WarpImg(It_BiRect,p, rect):

	x = np.linspace(rect[1]+p[1], rect[3]+p[1], rect[3]-rect[1])
	y = np.linspace(rect[0]+p[0], rect[2]+p[0], rect[2]-rect[0])
	[X, Y] = np.meshgrid(x, y)
	WarpedImg = It_BiRect.ev(X,Y)

	return WarpedImg
	
