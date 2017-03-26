import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from thresholding import *


# Read in an image and grayscale it
image = mpimg.imread('test_images/test5.jpg')

# Choose a Sobel kernel size
ksize = 3 # Choose a larger odd number to smooth gradient measurements

# S Chan thresholding
binary,s_chan = s_channel_threshold(image)

# Apply each of the thresholding functions
gradx = abs_sobel_thresh(image, 1,0, thresh=(20, 255))
grady = abs_sobel_thresh(image, 0,1, thresh=(20, 255))
mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(20, 255))
DELTA_ANGLE = np.pi/10
left_lane_dir = dir_threshold(image, sobel_kernel=5, thresh=(np.pi/4- DELTA_ANGLE, np.pi/4 + DELTA_ANGLE))
right_lane_dir = dir_threshold(image, sobel_kernel=5, thresh=( -np.pi/4- DELTA_ANGLE, -np.pi/4 + DELTA_ANGLE))

# Plot the result
def plotimg(ax,img,title=''):
    ax.imshow(img, cmap='gray')
    ax.set_axis_off()
    ax.set_title(title)

# Prepare subplots
f, ((ax1, ax2, ax3,ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2, 4, figsize=(24, 9))
f.tight_layout()


plotimg(ax1,image,'Original Image')
plotimg(ax2,s_chan,'S Channel')
plotimg(ax3,binary,'Thresholded S Channel')

combined = np.zeros_like(mag_binary)
combined[((gradx == 1) & (grady == 1))] = 1
plotimg(ax4,combined,'Grad. X & Y')

combined = np.zeros_like(mag_binary)
combined[(mag_binary == 1)] = 1
plotimg(ax5,combined,'Magn')

grad_mag = np.zeros_like(mag_binary)
grad_mag[((gradx == 1) & (grady == 1))  & (mag_binary == 1)] = 1
plotimg(ax6,grad_mag,'Grad. X & Y & Magn')

combined = np.zeros_like(mag_binary)
combined[ ((gradx == 1) & (grady == 1)) & ( (left_lane_dir == 1) | (right_lane_dir == 1) ) & (mag_binary == 1)] = 1
plotimg(ax7,combined,'All Gray combined')

final = np.zeros_like(mag_binary)
final[(binary == 1) | (grad_mag == 1)] = 1
plotimg(ax8,final,'S chan or Gray combined')

plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()