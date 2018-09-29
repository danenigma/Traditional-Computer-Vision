import numpy as np 
def alignChannels(red, green, blue):
	"""Given 3 images corresponding to different channels of a color image,
	compute the best aligned result with minimum abberations

	Args:
	red, green, blue - each is a HxW matrix corresponding to an HxW image

	Returns:
	rgb_output - HxWx3 color image output, aligned as desired"""

	bestGreenAlignment = None  
	minSSD = 1e9

	for horizontalShift in range(-30, 31 , 1):
		#shiftedGreen = np.roll(green, horizontalShift, axis=0)

		for virticalShift in range(-30, 31 , 1):

			shiftedGreen = np.roll(np.roll(green, horizontalShift, axis=0),
								   virticalShift, axis=1)
			shiftedSSD = np.sqrt(np.sum(np.square(shiftedGreen-red)))
		
			if shiftedSSD < minSSD:
				minSSD = shiftedSSD
				bestGreenAlignment = shiftedGreen
				
	bestBlueAlignment = None  
	minSSD = 1e9

	for horizontalShift in range(-30, 31 , 1):


		for virticalShift in range(-30, 31 , 1):

			shiftedBlue = np.roll(np.roll(blue,horizontalShift, axis=0),
										 virticalShift, axis=1)
			shiftedSSD = np.sqrt(np.sum(np.square(shiftedBlue-red)))

			if shiftedSSD < minSSD:
				minSSD = shiftedSSD
				bestBlueAlignment = shiftedBlue
				
	alignedImage = np.zeros([blue.shape[0], blue.shape[1], 3])

	alignedImage[:,:,0] = red
	alignedImage[:,:,1] = bestGreenAlignment
	alignedImage[:,:,2] = bestBlueAlignment 
	
	
	return alignedImage

