## Project: Extraction of Interest Points and Establishing Correspondences using Harris Corner Detector and SIFT Operator

## Overview
---
In this project we consider a pair of images of the same scene taken from different viewpoints
and extract interest points in each image of the pair and automatically establish the correspondence
between the interest points using two different methods:: 

* Harris Corner Detector coupled with the Normalized Cross Correlation (NCC) and Sum of
Squared Differences (SSD) metrics to establish correspondence between the extracted interest
points

* SIFT operator implemented in the open source OpenCV library


[//]: # (Image References)
[image1]: ./write_up_images/HarrisNCCpair1_sigma2p6.jpg "Image 1"
[image2]: ./write_up_images/HarrisSSDpair1_sigma2p6.jpg "Image 2"
[image3]: ./write_up_images/sift_output_pair1_sigma2p6.jpg "Image 3"


## Instructions to Run
---
The Harris Corner detector is implemented in HarrisCorner_Tejas_Pant.ipynb and SIFT operator is implemented in SIFTOperator_Tejas_Pant.ipynb

## Results using Harris Corner Detector with NCC Metric
---
![alt text][image1]

## Results using Harris Corner Detector with SSD Metric
---
![alt text][image2]

## Results using SIFT Operator
---
![alt text][image3]

The detailed results and discussion can be found in hw4_TejasPant.pdf 