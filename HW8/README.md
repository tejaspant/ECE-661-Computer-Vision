## Project: Camera Calibration using Zhang's Algorithm

## Overview
---
In this project we determine the intrinsic and extrinsic parameters of a camera using the Zhang's algorithm. Specifically,

* We first use the Canny edge detector to detect edges in the calibration pattern and then
use the Hough transform to construct hough lines and detect the corners in the calibration
pattern as intersection of the Hough lines.
* Next, from the detected corners, we calculate the homography between the world coordinates
and the image pixel coordinates and use the homography to estimate the intrinsic and extrinsic
parameters of the camera using Zhang's algorithm.
* Lastly, we reference the intrinsic and extrinsic parameters using the Levenberg-Marquardt algo-
rithm.

[//]: # (Image References)

[image1]: ./write_up_images/canny_edge_dataset1_image5.jpg "Image 1"
[image2]: ./write_up_images/hough_lines_dataset1_image5.jpg "Image 2"
[image3]: ./write_up_images/corners_dataset1_image5.jpg "Image 3"
[image2]: ./write_up_images/reproj_dataset2_img13_on_1_woLM.jpg "Image 4"
[image3]: ./write_up_images/reproj_dataset2_img13_on_1_withLM.jpg "Image 5"

## Dataset
---
The camera has been calibrated for two data sets of images which can be found in Dataset1 and Dataset2_Tejas

## Instructions to Run
---
The solution is implemented in the code hw7_TejasPant.m.

## Result
---
Here are some of the sample results:
![alt text][image1]
![alt text][image2]
![alt text][image3]

The effect of refining the intrinsic and extrinsic parameters using the Levenberg-Marquardt algorithm can be seen here where we reproject the corners from Images 13 in Dataset 2 into Image 1 of Dataset 2
Without Levenberg-Marquardt refinement
![alt text][image4]

With Levenberg-Marquardt refinement
![alt text][image5]
The detailed results and discussion can be found in hw8_TejasPant.pdf 