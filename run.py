import matplotlib.pyplot as plt
from thresholding import *
import glob
import pickle

"""
This is the main function to process the video applying all the required steps
"""

# Camera calibration
def calibrate():
    """
    Opens calibration image and find chessboard corners in image
    :return: calibration matrix & distance
    """

    # prepare object points
    nx = 9
    ny = 6

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((nx * ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob('camera_cal/calib*.jpg')

    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        print("Processing file " + fname)
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
            write_name = 'corners_found' + fname.split('/')[-1]
            cv2.imwrite('camera_cal/' + write_name, img)
            cv2.imshow(write_name, img)
            cv2.waitKey(500)
        else:
            print("Error: no chessboard corners found for " + fname)
            assert (False)

    cv2.destroyAllWindows()

    img_size = (img.shape[1], img.shape[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    pickle.dump( dist_pickle, open( "camera_cal/wide_dist_pickle.p", "wb" ) )

    return mtx,dist

def undistort(img,mtx,dist):
    """
    Distortion correction
    :param img: input image
    :param mtx: calibration matrix
    :param dist:
    :return: undistorded image
    """
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist

# Color/gradient threshold
def apply_thresholds(img_rgb):
    gradx = abs_sobel_thresh(img_rgb, 1,0, thresh=(20, 255))
    grady = abs_sobel_thresh(img_rgb, 0,1, thresh=(20, 255))
    mag_binary = mag_thresh(img_rgb, sobel_kernel=ksize, mag_thresh=(20, 255))
    binary,s_chan = s_channel_threshold(img_rgb)

    # Combine all these together:
    final = np.zeros_like(mag_binary)
    final[(binary == 1) | (((gradx == 1) & (grady == 1))  & (mag_binary == 1)) ] = 1

    return final

# Perspective transform
def transform_perspective(img):
    length, width = img.shape[1], img.shape[0]

    # Keep only interesting part of the image
    lb = (266,694)
    rb = (1167,694)
    lt = (588,437)
    rt = (740,437)

    selection_img = np.copy(img)
    cv2.line(selection_img, lb, lt, color=[0, 255, 0], thickness=5)
    cv2.line(selection_img, rt, rb, color=[0, 255, 0], thickness=5)

    offset = 100
    src = np.float32([lb, lt, rt, rb])
    #dst = np.float32([[width, offset], [offset, offset], [offset,length - offset], [width, length-offset]])
    dst = np.float32([[offset,width*2], [offset, offset], [length - offset,offset], [length - offset,width*2]])
    # use cv2.getPerspectiveTransform() to get M, the transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # use cv2.warpPerspective() to warp your image to a top-down view
    warped = cv2.warpPerspective(img, M, (length,width*2), flags=cv2.INTER_LINEAR)

    return warped, selection_img
# Detect lane lines
def detect_lines():
    return

# Determine the lane curvature
def find_lane_curvature():
    return

def display():
    return

def pipeline(img):
    return

# Plot the result
def plotimg(ax,img,title=''):
    ax.imshow(img, cmap='gray')
    ax.set_axis_off()
    ax.set_title(title)

with open('camera_cal/wide_dist_pickle.p', 'rb') as handle:
    calib = pickle.load(handle)
    mtx = calib["mtx"]
    dist = calib["dist"]

# Let's test the calibration
testimg = cv2.imread("camera_cal/invalid_1.jpg")
undistort_test = undistort(testimg,mtx,dist)
cv2.imwrite('camera_cal/undistort_test.jpg', undistort_test)
#cv2.imshow("Undistorded image", undistort_test)

images = glob.glob('test_images/*.jpg')
#for idx, fname in enumerate(images):
def show_pipeline(fname):
    img = cv2.imread(fname)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    undistorted = undistort(img_rgb,mtx,dist)
    warped, selection_img = transform_perspective(undistorted)
    #f, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2, 4)
    f,(ax1,ax2,ax3,ax4) = plt.subplots(1,4)
    plotimg(ax1,img_rgb,"Original image")
    plotimg(ax2,undistorted,"Undistorted image")
    plotimg(ax3,selection_img,"Selection used")
    plotimg(ax4,warped,"Wraped image")

    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show(block=True)
    #plt.show()

show_pipeline("test_images/test1.jpg")