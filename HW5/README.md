## Project: Image Mosaicing using SIFT Operator and RANSAC Algorithm

## Overview
---
In this project we carry out image mosaicing which consists of stitching together a sequence of
overlapping photos of a scene to create a single panoramic photo by: 

* First using the SIFT operator to extract interest points between the image pairs and establishing
correspondence between the interest points

* Next, the RANSAC algorithm is used to eliminate the false correspondences (outliers) between
the interest points and the homography between the image pairs is estimated using the noisy
true correspondences (inliers) with the Linear-Least Squares Method

* In the last step, a more accurate Nonlinear Least-Squares Method like the Levenberg-Marquardt
Method is used to determine a more precise homography using the homography estimated by
the Linear-Least Squares Method as an initial guess of the solution

[//]: # (Image References)
[image1]: ./write_up_images/imagepair23.jpg "Image 1"
[image2]: ./write_up_images/imagepair23_with_outliers.jpg "Image 2"
[image3]: ./write_up_images/panaromicimagewithoutlm.jpg "Image 3"
[image4]: ./write_up_images/variation_csot_iterations_LM.jpg "Image 4"

## Instructions to Run
---
The solution is implemented in the code h5_TejasPant.py.

## Results using SIFT and RANSAC for Outlier Rejection
---
![alt text][image1]
![alt text][image2]


## Results of Image Mosaicing
---
![alt text][image3]

The detailed results and discussion can be found in hw5_TejasPant.pdf 