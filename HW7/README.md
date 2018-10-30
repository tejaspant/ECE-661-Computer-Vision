## Project: Image Classifier using Local Binary Pattern (LBP) feature extraction and Nearest-Neighbour Classifier

## Overview
---
In this project we implement a simple image classifier using Local Binary Pattern (LBP) feature
extraction algorithm and a Nearest Neighbour (NN) classifier. Specifically,

* We first implement the LBP feature extraction algorithm to obtain a histogram of feature
vector for each image in the training set.
* Next, using the Euclidean distance metric in the feature vector space, we find the k-nearest
neighbours of each image in test set from the training set and assign the label to the test set
image which appears maximum number of times.
* In the last step, we construct a confusion matrix based on the classification results to calculate
the overall accuracy of the image classification algorithm.

[//]: # (Image References)

[image1]: ./write_up_images/Class_beach_ImageNum_0.png "Image 1"
[image2]: ./write_up_images/Class_building_ImageNum_0.png "Image 2"
[image3]: ./write_up_images/Class_car_ImageNum_0.png "Image 3"
[image4]: ./write_up_images/Class_mountain_ImageNum_0.png "Image 4"
[image5]: ./write_up_images/Class_tree_ImageNum_0.png "Image 5"
[image6]: ./write_up_images/confusion_matrix.png "Image 6"

## Training Data Set
---
The image classifier has been trained on 5 classes of images, Beach, Buildings, Car, Mountain and Tree. Each class has 20 images. 

## Test Data Set
---
The test set has 5 images for each class of images. The training and test set can be found in imagesDatabaseHW7

## Instructions to Run
---
The solution is implemented in the code hw7_TejasPant.py.

## Result
---
Here are some of the sample results:

<\bFig. 1: LBP Histogram for Sample 1 in Class Beach>
![alt text][image1]

Fig. 2: LBP Histogram for Sample 1 in Class Building
![alt text][image2]

Fig. 3: LBP Histogram for Sample 1 in Class Car
![alt text][image3]

Fig. 4: LBP Histogram for Sample 1 in Class Mountain
![alt text][image4]

Fig. 5: LBP Histogram for Sample 1 in Class Tree
![alt text][image5]

Fig. 6: Confusion matrix based on images in Test Set
![alt text][image6]

The overall accuracy of the classifier is 72% based on the confusion matrix.
The detailed results and discussion can be found in hw7_TejasPant.pdf 
