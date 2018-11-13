import numpy as np
import cv2

def rodrigues(r):
	# Replace pass by your implementation
	
	r = r.reshape(3, 1)	
	th  = np.linalg.norm(r)
	

	if th == 0:
		return np.eye(3), cv2.Rodrigues(r.astype('float'))[0]

	u = r/th

	a1, a2, a3 = u[0], u[1], u[2]
	 
	u_x = np.array([[0,  -a3, a2],
				   [a3,  0, -a1],
				   [-a2, a1, 0]])
	
		   
	#R = np.eye(3) + np.sin(th)*rx + (1 - np.cos(th) )*(r_n @ r_n.T - np.eye(3))
	I = np.eye(3).astype('float')
	R = I*np.cos(th) + (1 - np.cos(th))*(u @ u.T) + u_x*np.sin(th) 
	
	return R, cv2.Rodrigues(r.astype('float'))[0]
def invRodrigues(R):
	
	th = np.arccos((np.trace(R) - 1)/2)
	if th == 0 or np.isnan(th):
		return np.zeros(3).reshape(-1, 1), cv2.Rodrigues(R.astype('float'))[0]
	else:
		'''
		r = (1/(2*np.sin(th)))*np.array([R[2, 1] - R[1, 2],
			 							 R[0, 2] - R[2, 0],
		 								 R[1, 0] - R[0, 1]]).T			 		
		'''
		A = (R - R.T)/2
		a32, a13, a21 = A[2,1], A[0,2], A[1,0]
		
		rho = np.array([a32, a13, a21]).T
		s  = np.linalg.norm(rho)
		c  = (np.trace(R) - 1)/2
		u  = rho/s
		th = np.arctan2(s, c)
		r  = u*th
		
				 
	return r, cv2.Rodrigues(R.astype('float'))[0].reshape(-1)

if __name__ == '__main__':

	r = np.random.randint(0, 3, (3, 1))
	Rs = rodrigues(r)
	
	print('me: ', Rs[0])
	print('opencv: ', Rs[1])
	
