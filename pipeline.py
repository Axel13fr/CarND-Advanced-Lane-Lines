
from thresholding import *
from line_extraction import *
import pickle

# Camera calibration
class Calibration():
    def __init__(self):
        self.mtx = None
        self.dist = None
    def calibrate(self):
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

        self.mtx = mtx
        self.dist = dist
        return mtx,dist
    def load(self):
        with open('camera_cal/wide_dist_pickle.p', 'rb') as handle:
            calib = pickle.load(handle)
            self.mtx = calib["mtx"]
            self.dist = calib["dist"]


def undistort(img,calib):
    """
    Distortion correction
    :param img: input image
    :param mtx: calibration matrix
    :param dist:
    :return: undistorded image
    """
    undist = cv2.undistort(img, calib.mtx, calib.dist, None, calib.mtx)
    return undist

# Color/gradient threshold
def apply_thresholds(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gradx = abs_sobel_thresh(gray, 1,0, thresh=(20, 255))
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
    Minv = cv2.getPerspectiveTransform(dst,src)
    # Perspective transform
    warped = cv2.warpPerspective(img, M, (length,width), flags=cv2.INTER_LINEAR)

    if draw_lines:
        drawLinesFromPoints(d_lb,d_rb,d_lt,d_rt,warped)

    return warped, selection_img, Minv


def drawLinesFromPoints(p1, p2, p3, p4, img):
    cv2.line(img, p1, p2, color=[0, 255, 0], thickness=5)
    cv2.line(img, p2, p3, color=[0, 255, 0], thickness=5)
    cv2.line(img, p3, p4, color=[0, 255, 0], thickness=5)
    cv2.line(img, p4, p1, color=[0, 255, 0], thickness=5)


# Detect lane lines
def detect_lines(thresh_image):
    left_fit, right_fit, left_curverad, right_curverad = find_lines(thresh_image)
    return left_fit, right_fit, left_curverad, right_curverad

# Draw resulting lines back onto the image
def draw_result(warped, left_fit, right_fit, mtx, undist):
    # Compute points from lift fits
    ploty = np.linspace(0, warped.shape[0] - 1, warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, mtx, (undist.shape[1], undist.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    return result

class Pipeline():
    def __init__(self,debugView=True,usePrevLines=True):
        self.left_fit = None
        self.right_fit = None
        self.frame_nb = 0
        self.calibration = Calibration()
        self.calibration.load()
        self.debugView = debugView
        self.usePrevLines = usePrevLines

    def process(self,img_rgb):
        # Undistort image with calibration data
        undistorted = undistort(img_rgb,self.calibration)
        # Threshold and wrap
        thresholded,l_chan,s_chan = apply_thresholds(undistorted)
        warped_thresh, ignr,Minv = transform_perspective(thresholded)

        # Find lines and draw result to the image
        self.left_fit, self.right_fit, l_rad, r_ad,res_img = find_lines(warped_thresh,None,
                                                                        self.left_fit,self.right_fit)
        result = draw_result(warped_thresh, self.left_fit, self.right_fit, Minv, undistorted)

        # Write curvature info on image
        cv2.putText(result, "Left Rad.: " + str(l_rad) + " Right Rad.: " + str(r_ad),
                    (200, 100), cv2.FONT_HERSHEY_SIMPLEX, thickness=3,fontScale=1,color=[0,0,0])

        # Extra debug mode
        if self.debugView:
            cv2.putText(res_img, "Frame Nb: " + str(self.frame_nb),
                        (400, 100), cv2.FONT_HERSHEY_SIMPLEX, thickness=3, fontScale=1, color=[255, 255, 255])
            vis = np.concatenate((res_img, result), axis=0)
            self.frame_nb += 1
        else:
            vis = result

        if not self.usePrevLines:
            self.left_fit = None
            self.right_fit = None
        return vis
