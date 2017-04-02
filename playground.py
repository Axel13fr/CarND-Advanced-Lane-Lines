import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from thresholding import *


# Read in an image and grayscale it
image = mpimg.imread('test_images/test2.jpg')

# Choose a Sobel kernel size
ksize = 3  # Choose a larger odd number to smooth gradient measurements

# S Chan thresholding
binary,h_chan,l_chan,s_chan = s_channel_threshold(image)

# Apply each of the thresholding functions
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
gradx = abs_sobel_thresh(gray, 1,0, thresh=(20, 255),sobel_kernel=ksize)
gradx_lchan = abs_sobel_thresh(l_chan, 1,0, thresh=(20, 255),sobel_kernel=ksize)
grady = abs_sobel_thresh(gray, 0,1, thresh=(20, 255))
mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(20, 255))
DELTA_ANGLE = np.pi/10
left_lane_dir = dir_threshold(image, sobel_kernel=5, thresh=(np.pi/4- DELTA_ANGLE, np.pi/4 + DELTA_ANGLE))
right_lane_dir = dir_threshold(image, sobel_kernel=5, thresh=( -np.pi/4- DELTA_ANGLE, -np.pi/4 + DELTA_ANGLE))

# Plot the result
def plotimg(ax,img,title='',cmap='gray'):
    ax.imshow(img, cmap=cmap)
    ax.set_axis_off()
    ax.set_title(title)


# Prepare subplots
f, ((ax1, ax2, ax3,ax4), (ax5, ax6, ax7, ax8), (ax9, ax10, ax11, ax12)) = plt.subplots(3, 4, figsize=(24, 9))
f.tight_layout()

plotimg(ax1,image,'Original Image')
plotimg(ax2,h_chan,'H Channel')
plotimg(ax3,l_chan,'L Channel')
plotimg(ax4,s_chan,'S Channel')

plotimg(ax5,binary,'Thresholded S Channel')

grad_x = np.zeros_like(mag_binary)
grad_x[(gradx == 1)] = 1
plotimg(ax6,grad_x,'Grad. X')

grad_y = np.zeros_like(mag_binary)
grad_y[(grady == 1)] = 1
plotimg(ax7,grad_y,'Grad Y')

grad_mag = np.zeros_like(mag_binary)
grad_mag[(mag_binary == 1)] = 1
plotimg(ax8,grad_mag,'Magn')

combined = np.zeros_like(mag_binary)
combined[ ((gradx == 1) & (grady == 1)) & (mag_binary == 1)] = 1
plotimg(ax9,combined,'X & Y & Magn.')

plotimg(ax10,gradx_lchan,'Grad. X on L Channel')

color_binary = np.dstack((np.zeros_like(gradx),gradx,binary))
plotimg(ax11,color_binary*255,'S chan or X stacked')

plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()

