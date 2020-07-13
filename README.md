# Advanced Lane Finding Project Writeup

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

[image1]: ./output_images/calibration2.jpg "chessboard corners"
[image2]: ./output_images/undist_images.jpg "Undistort"
[image3]: ./output_images/undistort0.jpg "test_undistort"
[image4]: ./output_images/binary_overdrive.jpg "binary overdrive"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

# [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code is in IPython notebook located in ["parameter_tuning.ipynb"](./parameter_tuning.ipynb)

Since we are calibrating the camera using a 9 by 6 chessboard on a flat surface, I have created a variable called `objpoints` that will hold x and y coordination of this 9 by 6 matrix with z = 0, i.e. (0,0,0) ,(1,0,0),(0,1,0)...and so on. This `objpoints` would be the same for all calibration image using this chessboard.

Then I used `cv2.findChessboardCorners` to find all x and y coordinates of those chessboard corners in each calibration images that took in different angles.

![alt text][image1]

By using a for loop to scan through all the caliration images, I appended all image points found by `cv2.findChessboardCorners` into `imgpoints` and matching number of `objpoints`. 

Next, I used those points and used `cv2.calibratCamera` to obtained `mtx,dist` that can be used for `cv2.undistort`. Following images are the example of the undistorted images

![alt text][image2]

To simplify later video processing code, I created a .py script for camera calibration so I just call this function. The py script is at ["frame_process.py"](./frame_process.py)

In that script, I created a global variable `cali` that will hold `mtx,dist`. In `def undistort_img(img)`, I will make sure this calibration only done once when this scripted is called to save runtime.

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

![alt text][image3]

In the "parameter_tuning.ipynb", I printed out all the undistorted images for reference and they are also located at "./undistort_test_images"

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I have run rgb, hsv, and hls to every test images to see which color filter would work the best among all the images. In the end, I concluded S and L in HLS, combined with R in RGB would be the channels I would use to do the color filtering. 

I also experimented the magnitude, directional and absolute gradients to find out the proper threshold values. I tested out the runtime of those algorithm, and decided to implement a flag called `overdrive` to choose to run only absolute gradient or all gradient combined. The runtime difference is around 1s per frame and this will save a lot of time when processing a video.

The I used to combine those gradients is 
```python
if overdrive: 
        mag_binary = mag_threshold(gray,sobel_kernel=ksize,thresh= mag_thresh)
        dir_binary = dir_threshold(gray, sobel_kernel=ksize, thresh=dir_thresh)
        combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    else:
        combined[(gradx == 1) & (grady == 1)] = 1   

```
Following thresholds are applied:

| Item      | Treshold Range   | 
|:---------:|:----------------:| 
| H in HSL  | (23, 100)        |
| S in HSL  | (170, 255)       |
| R in RGB  | (220, 255)       |
| abs gradient  | (50, 200)    |
| mag gradient  | (60, 150)    |
| dir gradient  | (0.7,1.3)    |

After experiment with all the threshold values, I created a function called `image_process` to combining all the process and output an binary image. When `overdrive` flag is set, one of colored output (R:R in RGB; G:Color filter result;B:Combined Gradient Result) is like:

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  