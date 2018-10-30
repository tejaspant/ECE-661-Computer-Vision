#############################################
#Tejas Pant
#ECE 661 Computer Vision HW7
#29th Oct 2018
#############################################

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
import glob
import BitVector
import pickle
import os.path
from collections import Counter

#LBP Feature Extraction Algorithm 
def LBP(img, R, P): 
    rowmax, colmax = img.shape[0]-R, img.shape[1]-R                                     
    LBP_hist = {t:0 for t in range(P+2)}                                            

    for i in range(R,rowmax):                                                         
        for j in range(R,colmax):  

            binary_pattern = []
            #Generate Binary Pattern around each pixel                                                             
            for p in range(P):  

                #Get points in X direction                                                      
                del_u = R * math.cos(2*math.pi*p/P)   
                if abs(del_u) < 0.001: del_u = 0.0                                    
                u = i + del_u
                u_base = int(u)
                delta_u = u - u_base

                #Get points in Y direction
                del_v = R * math.sin(2*math.pi*p/P)  
                if abs(del_v) < 0.001: del_v = 0.0
                v = j + del_v                                          
                v_base = int(v)                                         
                delta_v = v - v_base 

                #Calculate grayscale using Bilinear Interpolation
                grayscale_p = BilinearInterp(img, delta_u, delta_v, u_base, v_base)

                if grayscale_p >= img[i][j]:                                     
                    binary_pattern.append(1)                                                 
                else:                                                                
                    binary_pattern.append(0)                                                 

            #Using Avi Kak's BitVector Module
            bv =  BitVector.BitVector( bitlist = binary_pattern )                            
            intvals_for_circular_shifts  =  [int(bv << 1) for _ in range(P)]          
            minbv = BitVector.BitVector( intVal = min(intvals_for_circular_shifts), size = P )                                               
            bvruns = minbv.runs()                                                     
            encoding = None

            #Generate LBP Histogram
            if len(bvruns) > 2:                                                       
                LBP_hist[P+1] += 1                                                    
                encoding = P+1                                                        
            elif len(bvruns) == 1 and bvruns[0][0] == '1':                           
                LBP_hist[P] += 1                                                     
                encoding = P                                                          
            elif len(bvruns) == 1 and bvruns[0][0] == '0':                            
                LBP_hist[0] += 1                                                      
                encoding = 0                                                          
            else:                                                                     
                LBP_hist[len(bvruns[1])] += 1                                         
                encoding = len(bvruns[1])                                            
                                                                         
    return LBP_hist

#Bilinear Interpolation to get grayscale value at pixel location
def BilinearInterp(img, delx, dely, x0, y0):
    if (delx < 0.001) and (dely < 0.001):
        grayscale_x = float(img[x0][y0])                     
    elif (dely < 0.001):
        grayscale_x = (1 - delx) * img[x0][y0] +  delx * img[x0+1][y0]
    elif (delx < 0.001):
        grayscale_x = (1 - dely) * img[x0][y0] +  dely * img[x0][y0+1]
    else:
        grayscale_x = (1-delx)*(1-dely)*img[x0][y0] + (1-delx)*dely*img[x0][y0+1]  + delx*dely*img[x0+1][y0+1]  + delx*(1-dely)*img[x0+1][y0]   
    return grayscale_x

#Calculate LBP Histogram for images in each class in the training set
def LBPHistogramForClass(img_class, img_num, R_, P_):
    images = []
    lbp_hist = []
    
    images = glob.glob('../imagesDatabaseHW7/training/{}'.format(img_class) + '/*.jpg')

    img = cv2.imread(images[img_num])
    image = np.zeros((img.shape[0],img.shape[1], img.shape[2]),dtype='uint8')
    image_gray = np.zeros((img.shape[0],img.shape[1]),dtype='uint8')
    print(image.shape)
    print(image_gray.shape)
    
    image = cv2.imread(images[img_num])
    image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    plt.imshow(image)
    plt.close()
    lbp_hist = LBP(image_gray, R_, P_)

    #Plot LBP histogram
    plt.bar(list(lbp_hist.keys()), lbp_hist.values(), color='b')
    plt.savefig('Class_{}'.format(img_class) + '_ImageNum_{}'.format(int(img_num)) + '.png')

    return lbp_hist

#Calculate LBP Histogram for images in the test set
def LBPHistogramForTestSet(img_class, img_num, R_, P_):
    images = []
    lbp_hist = []
    
    images = glob.glob('../imagesDatabaseHW7/testing/{}'.format(img_class) + '_*.jpg')

    img = cv2.imread(images[img_num])
    image = np.zeros((img.shape[0],img.shape[1], img.shape[2]),dtype='uint8')
    image_gray = np.zeros((img.shape[0],img.shape[1]),dtype='uint8')
    #print(image.shape)
    #print(image_gray.shape)
    
    image = cv2.imread(images[img_num])
    image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    plt.imshow(image)
    plt.close()
    lbp_hist = LBP(image_gray, R_, P_)

    #Plot LBP histogram
    plt.bar(list(lbp_hist.keys()), lbp_hist.values(), color='b')
    plt.savefig('Class_{}'.format(img_class) + '_ImageNum_{}'.format(int(img_num)) + '_TestSet.png')

    return lbp_hist

#Neearest Neighbor Classifier
def NearestNeighborClassifier(lbp_hist_test_obj, lbp_hist_train, nImgs, nTrainImgs, k, nClass):

    #Convert dictionary to an array
    lbp_hist_test = np.zeros((nImgs,10))
    for i in range(len(lbp_hist_test_obj)): #Image number in test set
        lbp_hist_test[i,:] = np.array(list(lbp_hist_test_obj[i].values()))

    euclid_dt = np.zeros((nImgs,lbp_hist_train.shape[0]))
    test_img_labels = np.zeros((nImgs,k),dtype='int')
    label = np.zeros(nImgs,dtype='int')
    for i in range(nImgs): #Number of images in test set
        for j in range(lbp_hist_train.shape[0]): #Total images in training set
            euclid_dt[i,j] = np.linalg.norm(lbp_hist_test[i,:]-lbp_hist_train[j,1:])
        euclid_dt_sort_idx = np.argsort(euclid_dt[i,:]) 
        euclid_dt_sort = np.sort(euclid_dt[i,:])
        #print(euclid_dt[i,:])
        #print(euclid_dt_sort_idx)

        for k_idx in range(k):
            if(euclid_dt_sort_idx[k_idx]<(nTrainImgs*1)):
                test_img_labels[i, k_idx] = 0
            elif (euclid_dt_sort_idx[k_idx]<(nTrainImgs*2)):
                test_img_labels[i, k_idx] = 1
            elif (euclid_dt_sort_idx[k_idx]<(nTrainImgs*3)):
                test_img_labels[i, k_idx] = 2
            elif (euclid_dt_sort_idx[k_idx]<(nTrainImgs*4)):
                test_img_labels[i, k_idx] = 3
            elif (euclid_dt_sort_idx[k_idx]<(nTrainImgs*5)):
                test_img_labels[i, k_idx] = 4

        #print(test_img_labels[i,:])
        #Get label with maximum appearence
        label[i],freq = Counter(list(test_img_labels[i,:])).most_common(1)[0] 
        #print(label[i])
        #print(freq)

    return label

if __name__ == "__main__":

    ImageClassificationStep = "Train" #Train or Predict

    numTrainImages = 20
    numClasses = 5
    numTestImages = 5

    #LBP Parameters
    R_lbp = 1 # Radius of circular pattern of neighbors for LBP Algorithm
    P_lbp = 8  # Number of points on the circular patter

    #k Nearest-Neighbor Classifier Parameters
    k_nn = 5

    if ImageClassificationStep == "Train":
        print("You have selected Training mode")
        lbp_hist_train_beach_list = []
        lbp_hist_train_build_list = []
        lbp_hist_train_car_list = []
        lbp_hist_train_mount_list = []
        lbp_hist_train_tree_list = []

        #Calculate LBP histogram for all training images of all classes
        for i in range(numTrainImages):

            print("\nGenerating LBP for Beach Image")
            lbp_hist_train_beach = LBPHistogramForClass("beach", i, R_lbp, P_lbp)
            lbp_hist_train_beach_list.append(lbp_hist_train_beach)
            print("LBP Generated")

            print("\nGenerating LBP for Building Image")
            lbp_hist_train_build = LBPHistogramForClass("building", i, R_lbp, P_lbp)
            lbp_hist_train_build_list.append(lbp_hist_train_build)
            print("LBP Generated") 

            print("\nGenerating LBP for Car Image")
            lbp_hist_train_car = LBPHistogramForClass("car", i, R_lbp, P_lbp)
            lbp_hist_train_car_list.append(lbp_hist_train_car)
            print("LBP Generated") 

            print("\nGenerating LBP for Mountain Image")
            lbp_hist_train_mount = LBPHistogramForClass("mountain", i, R_lbp, P_lbp)
            lbp_hist_train_mount_list.append(lbp_hist_train_mount)
            print("LBP Generated") 

            print("\nGenerating LBP for Tree Image")
            lbp_hist_train_tree = LBPHistogramForClass("tree", i, R_lbp, P_lbp)
            lbp_hist_train_tree_list.append(lbp_hist_train_tree)
            print("LBP Generated") 

        #Save LBP histogram for all class of training set images one-by-one
        filehandler = open("TrainSetLBP_Beach.obj","wb")
        pickle.dump(lbp_hist_train_beach_list,filehandler)
        filehandler.close()

        filehandler = open("TrainSetLBP_Buildings.obj","wb")
        pickle.dump(lbp_hist_train_build_list,filehandler)
        filehandler.close()

        filehandler = open("TrainSetLBP_Car.obj","wb")
        pickle.dump(lbp_hist_train_car_list,filehandler)
        filehandler.close()

        filehandler = open("TrainSetLBP_Mountain.obj","wb")
        pickle.dump(lbp_hist_train_mount_list,filehandler)
        filehandler.close()

        filehandler = open("TrainSetLBP_Tree.obj","wb")
        pickle.dump(lbp_hist_train_tree_list,filehandler)
        filehandler.close()

        print("Training Complete")

    elif ImageClassificationStep == "Predict":
        print("You have selected Prediction mode")

        if not os.path.exists("TrainSetLBP_Beach.obj"):
            print("The Image Classifier has not been trained on Beach Images")
            exit()

        if not os.path.exists("TrainSetLBP_Buildings.obj"):
            print("The Image Classifier has not been trained on Building Images")
            exit()

        if not os.path.exists("TrainSetLBP_Car.obj"):
            print("The Image Classifier has not been trained on Car Images")
            exit()

        if not os.path.exists("TrainSetLBP_Mountain.obj"):
            print("The Image Classifier has not been trained on Mountain Images")
            exit()

        if not os.path.exists("TrainSetLBP_Tree.obj"):
            print("The Image Classifier has not been trained on Tree Images")
            exit()

        #Load saved LBP histograms for all class of images one-by-one
        file1 = open("TrainSetLBP_Beach.obj",'rb')
        object_file1 = pickle.load(file1)
        file1.close()
        beach_hist = np.zeros((numTrainImages,10))
        for i in range(len(object_file1)):
            beach_hist[i,:] = np.array(list(object_file1[i].values()))

        file2 = open("TrainSetLBP_Buildings.obj",'rb')
        object_file2 = pickle.load(file2)
        file2.close()
        build_hist = np.zeros((numTrainImages,10))
        for i in range(len(object_file2)):
            build_hist[i,:] = np.array(list(object_file2[i].values()))

        file3 = open("TrainSetLBP_Car.obj",'rb')
        object_file3 = pickle.load(file3)
        file3.close()
        car_hist = np.zeros((numTrainImages,10))
        for i in range(len(object_file3)):
            car_hist[i,:] = np.array(list(object_file3[i].values()))

        file4 = open("TrainSetLBP_Mountain.obj",'rb')
        object_file4 = pickle.load(file4)
        file4.close()
        mount_hist = np.zeros((numTrainImages,10))
        for i in range(len(object_file4)):
            mount_hist[i,:] = np.array(list(object_file4[i].values()))

        file5 = open("TrainSetLBP_Tree.obj",'rb')
        object_file5 = pickle.load(file5)
        file5.close()
        tree_hist = np.zeros((numTrainImages,10))
        for i in range(len(object_file5)):
            tree_hist[i,:] = np.array(list(object_file5[i].values())) 

        hist_train = np.zeros((numTrainImages*numClasses,11))

        #Intialize confusion matrix
        confusion_mat = np.zeros((numClasses, numClasses), dtype = 'int')

        for i in range(numClasses):
            idx_s = numTrainImages * i
            idx_e = idx_s + numTrainImages
            hist_train[idx_s:idx_e,0] = i

        #Collect histograms all training set images
        hist_train[:,1:] = np.concatenate((beach_hist,build_hist,car_hist,mount_hist,tree_hist),axis=0)

        lbp_hist_test_beach_list = []
        lbp_hist_test_build_list = []
        lbp_hist_test_car_list = []
        lbp_hist_test_mount_list = []
        lbp_hist_test_tree_list = []

        for i in range(numTestImages):
            trueClassName = "beach"
            trueClass = 0
            lbp_hist_test_beach = LBPHistogramForTestSet(trueClassName, i, R_lbp, P_lbp)
            lbp_hist_test_beach_list.append(lbp_hist_test_beach)

        #NN-Classifier on all 5 images in test set
        predicted_idx = NearestNeighborClassifier(lbp_hist_test_beach_list, hist_train, numTestImages, numTrainImages, k_nn, numClasses)
        #Fill row of confusion matrix
        label_unique, label_unique_count = np.unique(predicted_idx, return_counts=True)
        confusion_mat[trueClass,label_unique] = label_unique_count

        for i in range(numTestImages):
            trueClassName = "building"
            trueClass = 1
            lbp_hist_test_build = LBPHistogramForTestSet(trueClassName, i, R_lbp, P_lbp)
            lbp_hist_test_build_list.append(lbp_hist_test_build)

        #NN-Classifier on all 5 images in test set
        predicted_idx = NearestNeighborClassifier(lbp_hist_test_build_list, hist_train, numTestImages, numTrainImages, k_nn, numClasses)
        #Fill row of confusion matrix
        label_unique, label_unique_count = np.unique(predicted_idx, return_counts=True)
        confusion_mat[trueClass,label_unique] = label_unique_count

        for i in range(numTestImages):
            trueClassName = "car"
            trueClass = 2
            lbp_hist_test_car = LBPHistogramForTestSet(trueClassName, i, R_lbp, P_lbp)
            lbp_hist_test_car_list.append(lbp_hist_test_car)

        #NN-Classifier on all 5 images in test set
        predicted_idx = NearestNeighborClassifier(lbp_hist_test_car_list, hist_train, numTestImages, numTrainImages, k_nn, numClasses)
        #Fill row of confusion matrix
        label_unique, label_unique_count = np.unique(predicted_idx, return_counts=True)
        confusion_mat[trueClass,label_unique] = label_unique_count

        for i in range(numTestImages):
            trueClassName = "mountain"
            trueClass = 3
            lbp_hist_test_mount = LBPHistogramForTestSet(trueClassName, i, R_lbp, P_lbp)
            lbp_hist_test_mount_list.append(lbp_hist_test_mount)

        #NN-Classifier on all 5 images in test set
        predicted_idx = NearestNeighborClassifier(lbp_hist_test_mount_list, hist_train, numTestImages, numTrainImages, k_nn, numClasses)
        #Fill row of confusion matrix
        label_unique, label_unique_count = np.unique(predicted_idx, return_counts=True)
        confusion_mat[trueClass,label_unique] = label_unique_count

        for i in range(numTestImages):
            trueClassName = "tree"
            trueClass = 4
            lbp_hist_test_tree = LBPHistogramForTestSet(trueClassName, i, R_lbp, P_lbp)
            lbp_hist_test_tree_list.append(lbp_hist_test_tree)

        #NN-Classifier on all 5 images in test set
        predicted_idx = NearestNeighborClassifier(lbp_hist_test_tree_list, hist_train, numTestImages, numTrainImages, k_nn, numClasses)
        #Fill row of confusion matrix
        label_unique, label_unique_count = np.unique(predicted_idx, return_counts=True)
        confusion_mat[trueClass,label_unique] = label_unique_count
        print("Confusion Matrix = ")
        print("")
        print(confusion_mat)





        


    
    
    
    
