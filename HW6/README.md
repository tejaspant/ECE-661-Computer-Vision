## Project: Image Segmentation using Otsu's Method

## Overview
---
In this project we perform image segmentation using the [Otsu's Method](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=4310076). Otsu's method identifies the optimum grayscale level that separates the foreground from the background. To carry out
image segmentation using Otsu's method we used two different approaches:

* In the first approach, **RGB based Image Segmentation**, we apply the Otsu's method
separately to the RGB channels of the image and then combine the three segmented
images to get the final result.
* In the second approach, **Texture based Segmentation**, we first use texture-based characterization of the pixels and then apply the Otsu's method to the different characterizations treating them as color channels.

[//]: # (Image References)
[image1]: ./write_up_images/light_house_rgb.jpg "Image 1"
[image2]: ./write_up_images/lighthouse_texture.jpg "Image 2"
[image3]: ./write_up_images/baby_rgb.jpg "Image 3"
[image4]: ./write_up_images/baby_texture.jpg "Image 4"
[image5]: ./write_up_images/ski_rgb.jpg "Image 5"
[image6]: ./write_up_images/ski_texture.jpg "Image 6"

## Instructions to Run
---
The solution is implemented in the code hw6_TejasPant.py.

## Results using RGB based Image Segmentation
---
![alt text][image1]
![alt text][image3]
![alt text][image5]

## Results using Texture based Image Segmentation
---
![alt text][image2]
![alt text][image4]
![alt text][image5]

The detailed results and discussion can be found in hw6_TejasPant.pdf 
