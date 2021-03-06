import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches

#my imports 
import cv2
from LucasKanade import LucasKanade




# write your script here, we recommend the above libraries for making your animation
frames = np.load('../data/carseq.npy')

H, W, T = frames.shape
rect = np.array([59, 116, 145, 151]).astype('float').T
p0 =  np.zeros(2)
carseqrects = np.zeros((T, 4))
carseqrects[0, :] = rect

fig=plt.figure(figsize=(4, 1))
columns = 5
rows = 1
index = 1

for i in range(1, T):

	p = LucasKanade(frames[:, :, i-1], frames[:, :, i], rect, p0=p0)
	frame = frames[:, :, i].copy()
	rect += np.array([p[1], p[0], p[1], p[0]]).T

	carseqrects[i, :] = rect
	
	
	cv2.rectangle(frame, (int(rect[0]),int(rect[1])), (int(rect[2]),int(rect[3])),(255), 1)
	if i in [2, 100, 200, 300, 400]:
		#cv2.imwrite('frame_'+str(i) + '.jpg', frame.astype('uint8'))
		rect_patch = patches.Rectangle((rect[0],rect[1]),
										rect[2]-rect[0],
										rect[3]-rect[1], 
										linewidth=1,edgecolor='g',facecolor='none')
		
		ax = fig.add_subplot(rows, columns, index)
		ax.add_patch(rect_patch )
		
		ax.imshow(frames[:, :, i], cmap='gray')
		ax.get_yaxis().set_visible(False)
		ax.get_xaxis().set_visible(False)

		index+=1

	cv2.imshow('input', frame)
	print('frame: ', i)
	if cv2.waitKey(10) == ord('q'):
		break
np.save('carseqrects.npy', carseqrects)
cv2.destroyAllWindows()
plt.show() 

