
---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

[video2]: ./project_output.mp4 "Video"
[inacurate]: ./report/inaccurate18.jpg "Innacurate calibration"
[distorded]: ./report/invalid_1.jpg "Before calibration correction"
[undistorded]: ./report/undistort_test.jpg "After calibration correction"
[thresholding]: ./report/thresholding.jpg "thresholding pipeline"
[thresholding2]: ./report/thresholding2.jpg "thresholding pipeline"
[pipeline]: ./report/pipeline_result.png "pipeline"
[pipeline2]: ./report/pipeline_result2.png "pipeline"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the calibration class code located in the file called `calibration.py`.  

In the calibrate() method, I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  The matrix and distance data are saved in a pickle file which can be later retrieved without calibrating again, using the load() method.

It is important to note that many provided samples are not correctly flat, leading to some bad corner matching results as in:

![inacurate]

Some other samples simply do not show the full chessboard, I therefore selected only 2samples to compute my calibration.
I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

Before 
![distorded]
After
![undistorded]

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding functions are in `thresholding.py`). They are used in apply_thresholds() located in  `piepline.py`. 
My conclusions were the following:
- direction threshold was too noisy
- y gradient picked too many features from the ground and some noise
- a combination of saturation channel to remain reliable when lighting conditions were changing was good
- x gradient best picked up line information on RGB channel. 
- L channel conveyed similar information to RGB so I didn't use it.

For examples in images, see in the section below along with the perspective transform.

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `transform_perspective()`, in the file `pipeline.py`.  The `transform_perspective()` function takes as inputs an image (`img`), and defines source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:
- lb = source left bottom, rt = source right top, d_rt = destination right top etc
- offset = 200 and width and length are the dimensions of the image

```
    lb = (130,width)
    rb = (1235,width)
    lt = (560,465)
    rt = (730,465)
    d_lb = (offset,width)
    d_rb = (offset, 0)
    d_lt = (length - offset, 0)
    d_rt = (length - offset, width)

```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 130, 720      | 200, 720        | 
| 1235, 720      | 200, 0      |
| 560, 465     | 1080, 0      |
| 730, 465      | 1080, 720        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

The complete projection and thresholding pipeline in action:
![pipeline][thresholding]
![pipeline][thresholding2]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Using the thresholded warped image, I use the find_lines() function located in `line_extraction.py`.
It consists of searching for the line starts at the bottom of the image using histogram peaks for the first image. It then searches for pixels in 9 boxes along the height of the image, 100px width each, giving a surface of about 4000pixels each. Each pixel in one of these box is used to fit a second order polynomial. If a box contains more than minpix pixels (500), its position is shifted to the average pixel position.

Starting from the example algorithm of the lesson, I had to tune the width of the boxes down to avoid picking up false positives aside the line main axis and I had to increase the minpix parameter so to avoid moving the boxes too much due to noise: 500 pixels is about 12% of a box which is not that much as a line section when cleanly detected takes about 50%.

Once I have already found lines, I simply use the previously found lines and search around it to find the updated linem using the same 50px margin.

Lastly, when one of the 2 line is too weak, meaning when it has less than 4300 pixels contributing to its polynomial fit, it is considered not enough to extrapolate it. So in that case and when the other line is strong (more than 4300pixels), I use the correct_fit_intersection() function which basically takes the strong line fit and transpose it to the detected starting point of the weak line. Basically, this ignore the weak line and use the polynomial fit of the strong line, just having it starting at the detected base point of the weak line.

You can see below and example of the confidence indicator I use as the sum of the pixels contributing to each lines. The red and blue surface are showing the margin area used to pick pixels to find the line fit.

![alt text][pipeline]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in calc_curvature() method in my code in `line_extraction.py`. It first converts all the picked points to fit a line from pixel coordinates into world coordinate in meters. After that, it fits again a 2nd degree polynomial using the transformed points and calculates the curvature from it for each lines.
The position of the vehicle with respect to the center is done in the same way:
- calculate the line start at the bottom of the image (base_x_left_m and base_x_right_m) by converting the image bottom from pixels to meters using the new previously calculated polynomial in world coordinates
- calculate the center of the line as (base_x_right_m + base_x_left_m) / 2
- calculate the distance between the center of the line and the center of the image (ie the middle of the image by transforming pixel width into world distance)

Note that the radius curvature is very sensible to small variations specially when going straight, it is therefore averaged over the last 10 samples for each line and then averaged again using the 2 curvature calculated for each line. Code for this is done in the line class in append_curv() method in `line_extraction.py` function.

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in get_line_fit_image() in my code in `line_extraction.py`.  This function is called in the find_lines() function. It computes a polygon along the found lines and colors the surface and the lines, then stack it onto the original image. It also generates an image made of a debug view showing the linefit and the projection of the found lines on the road.

![alt text][pipeline2]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_output.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

My approach managed to pass the project video without any average done onto the polynomial fits but this would have been something to do at least over 5 frames to smooth out the output better. 
If a line is not found at all in my implementation, this is no recovery mecanism implemented such as using the previous fit or if strong enough, using the other line fit as the 2 are parallel, just like I did when correcting the weak lines.
In case of a combination of faded lines (low contributing pixels count) and noise due to shade or damaged road for long enough, the average might not be enough to avoid loosing lines.
Weather is good in california but my approach would fail in case of rain or low lights due to: low gradients / low lightning / reflections on the water which can very much look like lines.

Tracking lines over time seems to be the next improvement to implement along with rejecting outliers based on the difference between found polynomial fit. Another approach can be to discard lines with an unrealistic curvature or high curvature variation based on road standards or even better GPS maps using the type of road with drive on. Only time is missing here :)

