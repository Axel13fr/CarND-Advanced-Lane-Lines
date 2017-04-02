import matplotlib.pyplot as plt
from thresholding import *
from line_extraction import *
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
    images = ["camera_cal/calibration2.jpg","camera_cal/calibration3.jpg" ]

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
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gradx = abs_sobel_thresh(gray, 1,0, thresh=(20, 255))
    grady = abs_sobel_thresh(gray, 0,1, thresh=(20, 255))
    mag_binary = mag_thresh(img_rgb, mag_thresh=(20, 255))
    binary, h_chan, l_chan, s_chan = s_channel_threshold(img_rgb)

    # Combine all these together:
    final = np.zeros_like(mag_binary)
    final[(binary == 1) | ((gradx == 1) & (mag_binary == 1)) ] = 1

    return final,l_chan,s_chan

# Perspective transform
def transform_perspective(img,draw_lines=False):
    length, width = img.shape[1], img.shape[0]

    # Define shape for perspective transformation
    lb = (130,width)
    rb = (1235,width)
    lt = (560,465)
    rt = (730,465)

    selection_img = np.copy(img)
    drawLinesFromPoints(lb, lt, rt, rb, selection_img)

    offset = 200
    src = np.float32([lb, lt, rt, rb])
    d_lb = (offset,width)
    d_rb = (offset, 0)
    d_lt = (length - offset, 0)
    d_rt = (length - offset, width)
    dst = np.float32([d_lb, d_rb, d_lt, d_rt])

    # get M, the transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Perspective transform
    warped = cv2.warpPerspective(img, M, (length,width), flags=cv2.INTER_LINEAR)

    if draw_lines:
        drawLinesFromPoints(d_lb,d_rb,d_lt,d_rt,warped)

    return warped, selection_img


def drawLinesFromPoints(p1, p2, p3, p4, img):
    cv2.line(img, p1, p2, color=[0, 255, 0], thickness=5)
    cv2.line(img, p2, p3, color=[0, 255, 0], thickness=5)
    cv2.line(img, p3, p4, color=[0, 255, 0], thickness=5)
    cv2.line(img, p4, p1, color=[0, 255, 0], thickness=5)


# Detect lane lines
def detect_lines(thresh_image):
    left_fit, right_fit, left_curverad, right_curverad = find_lines(thresh_image)
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


#mtx, dist =  calibrate()
# Let's test the calibration
testimg = cv2.imread("camera_cal/invalid_1.jpg")
undistort_test = undistort(testimg,mtx,dist)
cv2.imwrite('camera_cal/undistort_test.jpg', undistort_test)
#cv2.imshow("Undistorded image", undistort_test)

def show_pipeline(fname):
    img = cv2.imread(fname)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    undistorted = undistort(img_rgb,mtx,dist)
    warped, selection_img = transform_perspective(undistorted,draw_lines=True)
    thresholded,l_chan,s_chan = apply_thresholds(undistorted)
    warped_thresh, ignr = transform_perspective(thresholded)

    f, ((ax1,ax2,ax3), (ax4,ax5,ax6)) = plt.subplots(2, 3)
    plotimg(ax1,selection_img,"Selection used")
    plotimg(ax2,s_chan, "S Channel")
    plotimg(ax3,l_chan, "L Channel")
    plotimg(ax4,warped,"Wrapped RGB image")
    plotimg(ax5,thresholded,"Thresholded image")
    plotimg(ax6,warped_thresh,"Wrapped thresh. image")

    plt.suptitle(fname)
    plt.show(block=True)

def show_min_pipeline(fname):
    img = cv2.imread(fname)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    undistorted = undistort(img_rgb,mtx,dist)
    thresholded, l_chan, s_chan = apply_thresholds(undistorted)
    warped_thresh, ignr = transform_perspective(thresholded)
    f, (ax1,ax2,ax3) = plt.subplots(1,3)
    plotimg(ax1,img_rgb,"Selection used")
    plotimg(ax2, warped_thresh, "Wrapped thresh. image")

    left_fit, right_fit,left_curverad, right_curverad = find_lines(warped_thresh,ax3)

    plt.suptitle(fname + "Lft Cur " + str(left_curverad) + " Rght Cur " + str(right_curverad))
    plt.show()

#show_pipeline("test_images/straight_lines1.jpg")

images = glob.glob('test_images/*.jpg')
for idx, fname in enumerate(images):
    #show_pipeline(fname)
    show_min_pipeline(fname)