

## Advanced Lane Finding Project

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

[image10]: ./output_images/camera_calib.png "Camera_calib"
[image11]: ./output_images/undistore_example.png "Undist_exmp"
[image12]: ./output_images/region_2.png "Region"
[image13]: ./output_images/region_perspective.png "Region_persp"
[image14]: ./output_images/orig_img.png "Original_images"
[image15]: ./output_images/thresh_img.png "Trhesholded_images"
[image16]: ./output_images/poly_fit_1_2.png "Poly_fit"
[image17]: ./output_images/img_processed.png "Processed_img"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/571/view) individually and describe how I addressed each point in my implementation.

---

### Writeup / README

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the file 'camera_calib.py'. I implemented a class that store the chessboard parameters I used in calibration process.
These are: chessboard width, height, cam_calibration where result of calibration data exist, folder name where the chessboard
 images and a pickle file ('cam_calib.p') with result of calibration are stored.
Saving the result of calibration helps to avoid doing a calibration every time the programme is launched.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `obj_corners` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.
I then used the output `obj_points` and `img_points` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.
I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![Camera_calib][image10]

The "original image" represent the chessboard image used for camera calibration and "processed imaged" a distortion-corrected chessboard image.

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test image 'test2.jpg' from test images sample:

![Undist_exmp][image11]

How I did it? As I implemented already a class `CameraCalibration` which store all required data like a camera matrix and distortion coefficients I simply called a method `undistort` of this class to get distortion corrected image.
You can observe the distortion correction on the bushes on the hill that I marked with rectangles.

This process contains two main steps:
* use chessboard images to obtain image points and object points
* use the OpenCV functions cv2.calibrateCamera() and cv2.undistort() to compute the calibration and undistortion.


#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image. The functionality for color and gradient threshold can be found in `image_processor.py`. Here I implemented a class `BinaryImageCreator` having a parameter for kernel size I used for gradient threshold (Sobel operator).
The method `combi_treshold' is responsible does combination of color and gradient image thresholds. The methods `plot_image_list` and `plot_images` are helpful methods to plot images presenting of steps for the binary image processing.

Actually as suggested in the project lesson I provided a gradient threshold using:
* Sobel operator to take gradients in both the x and the y directions . Taking the gradient in the x direction emphasizes edges closer to vertical what in case of lane lines is good.
* magnitude of the gradients in x and Y direction
* direction of the gradients (the inverse tangent (arctangent) of the y gradient divided by the x gradient)

This is implemented in the method `sobel_treshold' in BinaryImageCreator class.

The color transform is provided in methods: `color_treshold_yellow` and `color_treshold_yellow'. Here I convert the imaged to HSL color space and seperate the saturation, hue and lightness.
 I used these channels to create binary image where only pixels having a H, S and L channel in the value range are provided.
 The range values are separtly defined for yellow and white color (detected alredy in the first project of this term.

At the end I combined the color transformation and gradients together. The result for all test images are shown bellow:

![Original_images][image14]

... and here the result images:

![Trhesholded_images][image15]

The detected edges around trees, cars and hills will be filtered later by applying of perspective transformation (region of interest).


#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

To performed a perspective transformation I implemented a class `RegionOfInterest` stored in the file "lane_tracker.py".  I didn't hardcoded source and destination points. Instead of that I provided parameters that defined percentage of the width and height of the region of interest depending on the image size.
I moved the bottom source up about some small distance as at front car hood is in bottom of every image a front, so the lane lines will never be visisble in this part of image.

I used source points `pts` and destination `dst` points to provide perspective matrix that and stored them in this class for future use (in the image pipline). The inverted matrix is provided here two.
The result of calculated coordinates of source points is presented on the image bellow:

![Region][image12]

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 560, 450      | 150, 0        |
| 720, 450      | 1130, 0      |
| 1280, 669.59     | 1130, 720      |
| 0, 669.59      | 150, 0        |

... and perspective matrix:

```
M [
 [  -0.459   -1.529  933.759]
 [  -0.      -1.966  884.521]
 [  -0.      -0.002    1.   ]]

M_inv [
 [   0.163   -0.778  535.51 ]
 [   0.      -0.509  450.   ]
 [  -0.      -0.001    1.   ]]
```

I verified that my perspective transform was working as expected by drawing the `pts` and `dst` points onto a test image "straight_lines1.jpg" and its warped counterpart to verify that the lines appear parallel in the warped image.

![Region_persp][image13]

Both lane lines stay parallel after persective transformation. Uff! not sure how it will work for images where lane lines are not straight on. Will see.

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I followed a suggestion in the lesson and use an implementation of lane lines detection per Sliding Windows and Fit a Polynomial. The functionality for this can be found in "lane_find_utils.py". This is source coded provided in code samples in the lesson.
Two most important functions are implemented here : `find_lane_points` and `find_lane_points_inc`.

* `find_lane_points` a histogram along all the columns in the lower half of the image and finds the bottom-most x position (or "base") of the left and right lane lines. The function then identifies 10 windows from which to identify lane pixels, each one centered on the midpoint of the pixels from the window below.
This effectively "follows" the lane lines up to the top of the binary image, and speeds processing by only searching for activated pixels over a small portion of the image. Pixels that belong to each lane line are identified. After that a second order polynomial is calculated which fit to to each lane lanes (Numpy.polyfit()) .
* The `find_lane_points_inc` function performs basically the same task, but searching of lines uses a lines found in previous video frame.

The image below demonstrates this process:

![Poly_fit][image16]

I didn't use a help class Line suggested in the lesson to receive the characteristics of each line detection.
 Instead of that I keep a history of line matching results which were evaluated as good one. A good lines  are those one which seems to be parallel to each other and the distance between them is not too big (much bigger then assumed lane width).

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The radius of the curvature of the lane is implemented in the function `calc_curve_radius` in file "lane_find_utils.
 I used a sample code provided in the lesson. I used a formula which allowed to get an approximation of the curve radius by dividing the first and second derivatives of the polynomial plus the part where the coordinate system is transferred from pixel to meter using X_METER_PER_PIXEL and Y_METER_PER_PIXEL factors.
 This calculation is based on the [article](https://www.intmath.com/applications-differentiation/8-radius-curvature.php).

Calculation a position of the vehicle with respect to center is implemented in the method `calc_car_offset` in lane_find_utils.py file.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in function `draw_lane_area` in lane_find_utils.py file. Here is an example of my result on a test image:

![Processed_img][image17]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

The implementation of pipline on single image is provided in the method `process` of the class `LaneDetector` in the file "lane_tracker.py".
I created two output videos:
* [project video](https://github.com/GosiaP/CarND-Advanced-Lane-Lines/blob/master/output_videos/project_video.mp4)
* [challange video](https://github.com/GosiaP/CarND-Advanced-Lane-Lines/blob/master/output_videos/challenge_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

After I used my pipline on "challenge_video.mp4" I realized that it is not perfect and there cup od stuff that could improved.

One of the is binary image creation. I base my color transformation of images on HLS color space. At first I tried to use only saturation channel as suggested in the lesson but this approach some plroblems for images with shadows. Then I went approach of using HSL channel in some low and ranges for yellow and white color separately. But it seems to be even this approach is not good enough for processing of images from "challenge_video". Maybe you, reviewer of my project, could you give some suggestions how I could approve in the future?

Another weakness of my image processing pipeline is defintion of region of interest that is used to calculate perspective transformation matrix. Any small changes the definition of source and destination points result in good or bad lane lanes points that doesn't become parallel enough.

I think those two problems I faced in my project have a very big impact on the no so good quality of lane recognition in "challenge_video"

