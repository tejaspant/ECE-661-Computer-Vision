#############################################
#Tejas Pant
#ECE 661 Computer Vision HW6
#21st Oct 2018
#############################################

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math

#Applying Otsu's algorithm on a grayscale image
def OtsuMethod(img, mask=None):
    #
    hist = np.zeros((256,1))
    pdf = np.zeros((256,1))
    out_img = np.zeros((img.shape[0],img.shape[1]),dtype='uint8')
    #Total number of pixels in the image    
    num_pixels = 0
    muT = 0.;
    thresh = -1;
    max_sigmab =-1;
    
    #Calculate histogram and number of pixels for each iteration
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            if mask is None or mask[i][j] !=0:
                muT = muT + img[i,j]
                num_pixels = num_pixels + 1
                hist[img[i,j]] = hist[img[i,j]] + 1
            
    #Get PDF from histogram
    pdf = hist / num_pixels

    #Mean grayscale level in image
    muT = muT / num_pixels

    #Sum of probabilities for levels lower than threhold level
    omgi = 0

    #Mean of grayscale levels lower than threshold grayscale level
    mui = 0

    for i in range(0, 256): 
        pi = pdf[i]
        omgi = omgi + pi
        mui = mui + i*pi
        
        #To prevent code from crashing
        if omgi == 0 or omgi==1:
            continue

        #Between Class Variance based on Eq. (18) in Otsu's paper
        sigmab = ((muT * omgi - mui)**2)/(omgi*(1-omgi))
        
        if sigmab > max_sigmab:
            thresh = i
            max_sigmab = sigmab
            
    #If no threhold found return black image
    if thresh == -1:
        return out_img
            
    #Generate output image
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            if img[i,j] > thresh:
                out_img[i,j]=255
                
    return out_img

def RGBSegment(img, img_name, aprch, iterations, invert):
    #Initialize output image
    out_img = np.zeros((img.shape[0],img.shape[1]),np.uint8)
    out_img.fill(255)
    
    #Iterations for the three color channels/ texture chanels of the image
    for ch_num in range(0,3):
        
        #Initial foreground mask
        mask_ch = None        
        
        #Iterations of Otsu's Algorithm
        for iter in range(iterations[ch_num]):
            mask_ch = OtsuMethod(img[:,:,ch_num],mask_ch) #imread reads RGB as BGR
            #Write the foreground mask for each channel for each Otsu's Algorithm Ieration
            cv2.imwrite(img_name + '_Channel{}'.format(int(ch_num)) + '_Iter{}'.format(int(iter)) + '_Ap{}'.format(aprch) + '.jpg',mask_ch)

        #Generate Combined Mask
        if invert[ch_num] == 1:
            out_img = cv2.bitwise_and(out_img,cv2.bitwise_not(mask_ch))
        else:
            out_img=cv2.bitwise_and(out_img, mask_ch)
    
    return out_img
  
#Generate texture based representation of image based on different window sizes
def getImageTexture(img, winsize):
 
    #Initialize texture image
    out_img = np.zeros((img.shape[0],img.shape[1],len(winsize)),np.uint8)

    #Calculate variance at each pixel location based on 3 different window sizes
    for wnum, wsize in enumerate(winsize):
        num_bcell = wsize/2
        for i in range(int(num_bcell), img.shape[0] - int(num_bcell)):
            for j in range(int(num_bcell), img.shape[1]- int(num_bcell)):
                out_img[i,j,int(wnum)] = np.int(np.var(img[i-int(num_bcell):i+int(num_bcell)+1,j-int(num_bcell):j-int(num_bcell)+1]))
    return out_img
    
#The contour extraction algorithm. This is done after segmentation  
def getForegroundContour(img):
    
    #Initialize final contour image
    out_img = np.zeros((img.shape[0],img.shape[1]),dtype='uint8')
    
    #Mark border if pixel value is not 0 AND one of 8-neighbor is 0
    for i in range(1, img.shape[0]-1):
        for j in range(1, img.shape[1]-1):
            if img[i,j]!=0 and np.min(img[i-1:i+2,j-1:j+2])==0:
                out_img[i,j] = 255
            
    return out_img

if __name__ == "__main__":
    
    approach = 1 # 1 = RGB based Segmentation, 2  = Texture based Segementation
    image_select = 1 # 1 = Lighthouse, 2 = Baby, 3 = Jumping Man

    if image_select == 1:
        image = cv2.imread('../HW6Pics/lighthouse.jpg')
        image_name = 'lighthouse'
        if approach == 1:
            niterChannels = [1,1,1]
            channelInvert = [1,1,0] #flag to indicate if need to invert mask
        elif approach == 2:
            niterChannels = [1,1,1]
            channelInvert = [1,1,1] #flag to indicate if need to invert mask       
    elif image_select == 2:
        image = cv2.imread('../HW6Pics/baby.jpg')
        image_name = 'baby'
        if approach == 1:
            niterChannels = [2,2,2]
            channelInvert = [1,1,1]
        elif approach == 2:
            niterChannels = [1,1,1]
            channelInvert = [1,1,1]
    elif image_select == 3:
        image = cv2.imread('../HW6Pics/ski.jpg')
        image_name = 'ski'
        if approach == 1:
            niterChannels = [1,2,1]
            channelInvert = [1,1,1]
        elif approach == 2:
            niterChannels = [1,1,1]
            channelInvert = [1,1,1]

    if approach == 1:
        output_image = RGBSegment(image,image_name,approach,niterChannels,channelInvert)
    elif approach == 2:
        #The image is converted to gray scale
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        window_size = [3,5,7]
        image_texture = getImageTexture(image_gray,window_size)
        output_image = RGBSegment(image_texture,image_name,approach,niterChannels,channelInvert)
    
    cv2.imwrite(image_name + '_ap{}'.format(int(approach)) + '_segmented_with_noise.jpg',output_image)
    
    struct_elem = np.ones((2,2),np.uint8)
    output_image = cv2.dilate(output_image,struct_elem)
    output_image = cv2.erode(output_image,struct_elem)   
    
    #Steps to remove noise from background
    struct_elem = np.ones((4,4),np.uint8)
    output_image=cv2.erode(output_image,struct_elem)
    output_image=cv2.dilate(output_image,struct_elem)    

    cv2.imwrite(image_name + '_ap{}'.format(int(approach)) + '_segmented_without_noise.jpg',output_image)
    
    #Extracting the contour
    outputcontour = getForegroundContour(output_image)
    
    cv2.imwrite(image_name+ '_ap{}'.format(int(approach)) + '_extracted_contour.jpg', outputcontour)
    
    
    
