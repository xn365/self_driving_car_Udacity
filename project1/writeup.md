# **Finding Lane Lines on the Road** 

## Writeup

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the images
* Test pipeline on test images
* Improve pipeline to suit Test/challenge video

---

### Reflection

### 1. My pipeline. 

My pipeline consisted of 9 steps: 
* Select yellow and white pixels
* Get gray image
* Apply Gaussian smoothing
* Apply canny
* Select interest region
* Find lines in hough space
* Separate line segments by their slope to left lines and right lines
* Polyfit left lines to one line, Polyfit right lines to one line
* Draw left line and right line on raw image
[//]: # (Image References)

[image1]: ./test_images_output/solidWhiteCurve_weighted.jpg "White Curve"
[image2]: ./test_images_output/solidWhiteRight_weighted.jpg "White right"
[image3]: ./test_images_output/solidYellowCurve2_weighted.jpg "Yellow Curve"
![alt text][image1]
To suit challenge video,I modified the draw_lines() function:
* Dump left line and right line in last frame.
* Find lines in current frame
* Compute lines slope
* Keep lines only it's slope change form last left/right line is smaller than delta_slope_threshold(set to 10 degree).
* Polyfit new left line and right line if they exist, use the last line else.
* Find the intersection of left line and right line
* Set y_max = y_intersection + 20
* Set y_min = image.shape[0]
* Draw lines on raw image


### 2. Identify potential shortcomings with your current pipeline

* Curved lane is not recognized
* Multiple lanes are not recognized
* The processing speed is slow
* Unable to work when light is dim

### 3. Suggest possible improvements to your pipeline

* Curve fitting instead of line fitting
* 

