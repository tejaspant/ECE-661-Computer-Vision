## Project: Face Recognition using Principal Component Analysis (PCA) and Linear Discriminant Analysis (LDA) and Object Detection using Cascaded Adaboost Algorithm

## Overview
---
In this project we perform two tasks. In the first task we perform face recognition in the following
manner,
* In the first step we carry out a Principal Component Analysis (PCA) and Linear Discrimi-
nant Analysis (LDA) of the images to transform the high-dimensional image data to a low-
dimensional manifold
* Next, we use a nearest-neighbor classifier to classify the face data

In the second task of this project we design an object detector (car detector) with an arbi-
trarily defined false positive rate using the cascaded Adaboost algorithm

[//]: # (Image References)

[image1]: ./write_up_images/facepix.png "Image 1"
[image2]: ./write_up_images/pca_versus_lda.png "Image 2"
[image3]: ./write_up_images/mean_image.png "Image 3"
[image4]: ./write_up_images/fpr_during_testing.jpg "Image 4"
[image5]: ./write_up_images/positive_car.png "Image 5"
[image6]: ./write_up_images/negative_Car.png "Image 6"

## Dataset
---
### Face Recognition Dataset: 
The dataset used for the face recognition study is the [Facepix](https://cubic.asu.edu/content/facepix-database) dataset. Here is a sample image from the dataset
![alt text][image1]

### Car Detection Dataset: 
The dataset used for the object detection study is obtained from a private entity and cannot be shared. Here is a sample image from the dataset

Positive
![alt text][image5]

Negative
![alt text][image6]

## Instructions to Run
---
The solution for the face recognition task is implemented in hw10_TejasPant_PCA_LDA.ipynb.
The solution for the object detection task is implemented in hw10_TejasPant_Adaboost.m.

## Result
---
### Face Recognition Study
Mean image from the dataset
![alt text][image3]

Comparison of Performance of PCA with LDA
![alt text][image2]

### Object Detection Study
Effect of number of stages in cascaded Adaboost classifier on false positive rate 
![alt text][image4]

The detailed results and discussion can be found in hw10_TejasPant.pdf 
