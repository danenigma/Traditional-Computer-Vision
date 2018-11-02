import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import animation
import matplotlib.patches as patches
from scipy.ndimage import correlate, convolve
from mpl_toolkits.mplot3d import axes3d, Axes3D

img = np.load('lena.npy')

# template cornes in image space [[x1, x2, x3, x4], [y1, y2, y3, y4]]
pts = np.array([[248, 292, 248, 292],
                [252, 252, 280, 280]])

# size of the template (h, w)
dsize = np.array([pts[1, 3] - pts[1, 0] + 1,
                  pts[0, 1] - pts[0, 0] + 1])

# set template corners
tmplt_pts = np.array([[0, dsize[1]-1, 0, dsize[1], -1],
                      [0, 0, dsize[0] - 1, dsize[0] - 1]])

def getOptimalg(X, y, lamda):
	S = np.matmul(X, X.T)
	return np.matmul(np.linalg.inv(S + lamda*np.eye(S.shape[-1])), np.matmul(X, y))
# apply warp p to template region of img
def imwarp(p):
    global img, dsize
    return img[p[1]:(p[1]+dsize[0]), p[0]:(p[0]+dsize[1])]


# get positive example
gnd_p = np.array([252, 248])  # ground truth warp
x = imwarp(gnd_p)  # the template

# stet up figure
fig, axarr = plt.subplots(1, 3)
axarr[0].imshow(img, cmap=plt.get_cmap('gray'))
patch = patches.Rectangle((gnd_p[0], gnd_p[1]), dsize[1], dsize[0],
                          linewidth=1, edgecolor='r', facecolor='none')
axarr[0].add_patch(patch)
axarr[0].set_title('Image')

cropax = axarr[1].imshow(x, cmap=plt.get_cmap('gray'))
axarr[1].set_title('Cropped Image')

dx = np.arange(-np.floor(dsize[1]/2), np.floor(dsize[1]/2)+1, dtype=int)
dy = np.arange(-np.floor(dsize[0]/2), np.floor(dsize[0]/2)+1, dtype=int)
[dpx, dpy] = np.meshgrid(dx, dy)
dpx = dpx.reshape(-1, 1)
dpy = dpy.reshape(-1, 1)
dp = np.hstack((dpx, dpy))
N = dpx.size

all_patches = np.ones((N*dsize[0], dsize[1]))
all_patchax = axarr[2].imshow(all_patches, cmap=plt.get_cmap('gray'),
                              aspect='auto', norm=colors.NoNorm())
axarr[2].set_title('Concatenation of Sub-Images (X)')

X = np.zeros((N, N))
Y = np.zeros((N, 1))

sigma = 5


def init():
    return [cropax, patch, all_patchax]


def animate(i):
	global X, Y, dp, gnd_p, sigma, all_patches, patch, cropax, all_patchax, N

	if i < N:  # If the animation is still running
		xn = imwarp(dp[i, :] + gnd_p)
		X[:, i] = xn.reshape(-1)
		Y[i] = np.exp(-np.dot(dp[i, :], dp[i, :])/sigma)
		all_patches[(i*dsize[0]):((i+1)*dsize[0]), :] = xn
		cropax.set_data(xn)
		all_patchax.set_data(all_patches.copy())
		all_patchax.autoscale()
		patch.set_xy(dp[i, :] + gnd_p)
		return [cropax, patch, all_patchax]
	else:  # Stuff to do after the animation ends
		fig3d = plt.figure()
		ax3d = fig3d.add_subplot(111, projection='3d')
		ax3d.plot_surface(dpx.reshape(dsize), dpy.reshape(dsize),
		                 Y.reshape(dsize), cmap=plt.get_cmap('coolwarm'))

		# Place your solution code for question 4.3 here
		#lamda = 0.
		#g = getOptimalg(X, Y, lamda)
		#print(g)
		columns = 2
		rows = 1
		#X, y = get_X_y()
		lamda = 0.
		g = getOptimalg(X, Y, lamda).reshape(29, 45)
		fig1 = plt.figure(figsize=(2, 1))

		ax = fig1.add_subplot(rows, columns, 1)
		ax.imshow(g)
		ax.get_yaxis().set_visible(False)
		ax.get_xaxis().set_visible(False)
		corr_image0 = correlate(img, g)

		ax = fig1.add_subplot(rows, columns, 2)
		ax.imshow(corr_image0)
		ax.get_yaxis().set_visible(False)
		ax.get_xaxis().set_visible(False)
		fig1.suptitle('Visualization of g and its response with lambda = 0. using correlation', fontsize=16)

		lamda = 1.
		g = getOptimalg(X, Y, lamda).reshape(29, 45)
		fig2 = plt.figure(figsize=(2, 1))

		ax = fig2.add_subplot(rows, columns, 1)
		ax.imshow(g)
		ax.get_yaxis().set_visible(False)
		ax.get_xaxis().set_visible(False)
		corr_image1 = correlate(img, g)

		ax = fig2.add_subplot(rows, columns, 2)
		ax.imshow(corr_image1)
		ax.get_yaxis().set_visible(False)
		ax.get_xaxis().set_visible(False)
		fig2.suptitle('Visualization of g and its response with lambda = 1. using correlation', fontsize=16)
		#convolution

		lamda = 0.
		g = getOptimalg(X, Y, lamda).reshape(29, 45)
		fig3 = plt.figure(figsize=(2, 1))

		ax = fig3.add_subplot(rows, columns, 1)
		ax.imshow(g)
		ax.get_yaxis().set_visible(False)
		ax.get_xaxis().set_visible(False)
		conv_image0 = convolve(img, g)

		ax = fig3.add_subplot(rows, columns, 2)
		ax.imshow(conv_image0)
		ax.get_yaxis().set_visible(False)
		ax.get_xaxis().set_visible(False)
		fig3.suptitle('Visualization of g and its response with lambda = 0. using convolution', fontsize=16)

		lamda = 1.
		g = getOptimalg(X, Y, lamda).reshape(29, 45)
		fig4 = plt.figure(figsize=(2, 1))

		ax = fig4.add_subplot(rows, columns, 1)
		ax.imshow(g)
		ax.get_yaxis().set_visible(False)
		ax.get_xaxis().set_visible(False)
		conv_image1 = convolve(img, g)

		ax = fig4.add_subplot(rows, columns, 2)
		ax.imshow(conv_image1)
		ax.get_yaxis().set_visible(False)
		ax.get_xaxis().set_visible(False)
		fig4.suptitle('Visualization of g and its response with lambda = 1. using convolution', fontsize=16)


		fig5 = plt.figure()

		lamda = 1.
		g = getOptimalg(X, Y, lamda).reshape(29, 45)
		conv_image_corrected = convolve(img, np.flipud(np.fliplr(g)))
		ax = fig5.add_subplot(1, 1, 1)
		ax.imshow(conv_image_corrected)
		ax.get_yaxis().set_visible(False)
		ax.get_xaxis().set_visible(False)
		fig5.suptitle('response with lambda = 1. convolution corrected via numpy indexing ', fontsize=16)

		plt.show()
		return []

def get_X_y():
	global X, Y, dp, gnd_p, sigma, N
	i = 0
	while i < N:  # If the animation is still running
		xn = imwarp(dp[i, :] + gnd_p)
		X[:, i] = xn.reshape(-1)
		Y[i] = np.exp(-np.dot(dp[i, :], dp[i, :])/sigma)
		i += 1
	return X, Y


# Start the animation
ani = animation.FuncAnimation(fig, animate, frames=N+1,
                              init_func=init, blit=True,
                              repeat=False, interval=10)
plt.show()

'''
columns = 2
rows = 1

X, y = get_X_y()

lamda = 0.
g = getOptimalg(X, Y, lamda).reshape(29, 45)
fig1 = plt.figure(figsize=(2, 1))

ax = fig1.add_subplot(rows, columns, 1)
ax.imshow(g)
ax.get_yaxis().set_visible(False)
ax.get_xaxis().set_visible(False)
corr_image0 = correlate(img, g)

ax = fig1.add_subplot(rows, columns, 2)
ax.imshow(corr_image0)
ax.get_yaxis().set_visible(False)
ax.get_xaxis().set_visible(False)
fig1.suptitle('Visualization of g and its response with lambda = 0.', fontsize=16)

lamda = 1.
g = getOptimalg(X, Y, lamda).reshape(29, 45)
fig2 = plt.figure(figsize=(2, 1))

ax = fig2.add_subplot(rows, columns, 1)
ax.imshow(g)
ax.get_yaxis().set_visible(False)
ax.get_xaxis().set_visible(False)
corr_image1 = correlate(img, g)

ax = fig2.add_subplot(rows, columns, 2)
ax.imshow(corr_image1)
ax.get_yaxis().set_visible(False)
ax.get_xaxis().set_visible(False)
fig2.suptitle('Visualization of g and its response with lambda = 1.', fontsize=16)
#convolution

lamda = 0.
g = getOptimalg(X, Y, lamda).reshape(29, 45)
fig3 = plt.figure(figsize=(2, 1))

ax = fig3.add_subplot(rows, columns, 1)
ax.imshow(g)
ax.get_yaxis().set_visible(False)
ax.get_xaxis().set_visible(False)
conv_image0 = convolve(img, g)

ax = fig3.add_subplot(rows, columns, 2)
ax.imshow(conv_image0)
ax.get_yaxis().set_visible(False)
ax.get_xaxis().set_visible(False)
fig3.suptitle('Visualization of g and its response with lambda = 0. using convolution', fontsize=16)

lamda = 1.
g = getOptimalg(X, Y, lamda).reshape(29, 45)
fig4 = plt.figure(figsize=(2, 1))

ax = fig4.add_subplot(rows, columns, 1)
ax.imshow(g)
ax.get_yaxis().set_visible(False)
ax.get_xaxis().set_visible(False)
conv_image1 = convolve(img, g)

ax = fig4.add_subplot(rows, columns, 2)
ax.imshow(conv_image1)
ax.get_yaxis().set_visible(False)
ax.get_xaxis().set_visible(False)
fig4.suptitle('Visualization of g and its response with lambda = 1. using convolution', fontsize=16)


fig5 = plt.figure()

lamda = 1.
g = getOptimalg(X, Y, lamda).reshape(29, 45)
conv_image_corrected = convolve(img, np.flipud(np.fliplr(g)))
ax = fig5.add_subplot(1, 1, 1)
ax.imshow(conv_image_corrected)
ax.get_yaxis().set_visible(False)
ax.get_xaxis().set_visible(False)
fig5.suptitle('response with lambda = 1. convolution corrected via numpy indexing ', fontsize=16)

plt.show()
'''

