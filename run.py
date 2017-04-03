import matplotlib.pyplot as plt
from pipeline import *
from line_extraction import *
import glob
import pickle

"""
This is the main function to process the video applying all the required steps
"""

# Plot the result
def plotimg(ax,img,title=''):
    ax.imshow(img, cmap='gray')
    ax.set_axis_off()
    ax.set_title(title)

def show_detailed_pipeline(fname):
    img = cv2.imread(fname)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    undistorted = undistort(img_rgb,mtx,dist)
    # DISPLAY ONLY
    warped, selection_img,Minv = transform_perspective(undistorted,draw_lines=True)

    thresholded,l_chan,s_chan = apply_thresholds(undistorted)
    warped_thresh, ignr,Minv = transform_perspective(thresholded)

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
    warped_thresh, ignr,Minv = transform_perspective(thresholded)
    f, (ax1,ax2,ax3) = plt.subplots(1,3)
    plotimg(ax1,img_rgb,"Selection used")
    plotimg(ax2, warped_thresh, "Wrapped thresh. image")

    left_fit, right_fit,left_curverad, right_curverad = find_lines(warped_thresh,ax3)

    plt.suptitle(fname + "Lft Cur " + str(left_curverad) + " Rght Cur " + str(right_curverad))
    plt.show()

def show_pipeline(fname):
    img = cv2.imread(fname)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pipeline = Pipeline(debugView=True,usePrevLines=False)
    result = pipeline.process(img_rgb)
    plt.imshow(result)
    plt.title("Pipeline Result")
    plt.show()

def run_example_images():
    images = glob.glob('test_images/*.jpg')
    for idx, fname in enumerate(images):
        #show_pipeline(fname)
        #show_min_pipeline(fname)
        show_pipeline(fname)

def run_video():
    from moviepy.editor import VideoFileClip
    output = 'project_output.mp4'
    clip2 = VideoFileClip('project_video.mp4')

    pipeline = Pipeline(debugView=True,usePrevLines=True)
    challenge_clip = clip2.fl_image(pipeline.process)
    challenge_clip.write_videofile(output, audio=False)

calib = Calibration()
#calib.calibrate()
calib.load()
# Let's test the calibration
testimg = cv2.imread("camera_cal/invalid_1.jpg")
undistort_test = undistort(testimg,calib)
cv2.imwrite('camera_cal/undistort_test.jpg', undistort_test)
#cv2.imshow("Undistorded image", undistort_test)

#run_video()
run_example_images()