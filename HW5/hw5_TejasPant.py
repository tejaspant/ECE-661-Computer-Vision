#############################################
#Tejas Pant
#ECE 661 Computer Vision HW5
#7th Oct 2018
#############################################

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import os
import math
import sys

def SIFTDetectInterestPoints(img_gray):
    sift = cv2.xfeatures2d.SIFT_create(nfeatures=5000,nOctaveLayers=4,contrastThreshold=0.03,edgeThreshold=10,sigma=4)
    kp, des = sift.detectAndCompute(img_gray,None)
    return kp, des

def BruteForceMatcher(des1, des2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)
    pts = []
    
    for a, b in matches:
        if a.distance < 0.75 * b.distance:
            pts.append([a])
        
    return pts

def displayImagewithInterestPoints(img1,img2,corners):
    #Get shape of the output image
    nrows = max(img1.shape[0], img2.shape[0])
    ncol = img1.shape[1]+img2.shape[1]
    
    #Initialize combined output image
    out_img = np.zeros((nrows,ncol,3))
    
    #Copy Image 1 to left half of the output image
    out_img[:img1.shape[0], :img1.shape[1]] = img1
    
    #Copy Image 2 to right half of the output image
    out_img[:img2.shape[0], img1.shape[1]:img1.shape[1]+img2.shape[1]] = img2

    for xy in corners:
        cv2.circle(out_img,(int(xy[0]), int(xy[1])),2,(255,0,0),2) #interest points from Image 1
        cv2.circle(out_img,(img1.shape[1]+int(xy[2]),int(xy[3])),2,(255,0,0),2) #Interest points form Image 2
        cv2.line(out_img,(int(xy[0]),int(xy[1])),(img1.shape[1]+int(xy[2]),int(xy[3])), (0,255,0)) #Lines joining interest points
    return out_img

def getXYCoordinateInteresPoints(matches,ip1_loc,ip2_loc):
    #Getting XY coordinates of corresponding Interest Points
    ip1_xy = []
    ip2_xy = []
    for match in matches:
        img1_idx = match[0].queryIdx
        img2_idx = match[0].trainIdx
        (x1,y1) = ip1_loc[img1_idx].pt
        (x2,y2) = ip2_loc[img2_idx].pt
        ip1_xy.append((x1, y1))
        ip2_xy.append((x2, y2))

    ip1_xy = np.array(ip1_xy)
    ip2_xy = np.array(ip2_xy)
    XY_Corres_IP = np.concatenate((ip1_xy,ip2_xy),axis=1)
    return XY_Corres_IP

# Finding Homography
def LinearLeastSquaresHomography(src_pts,dest_pts):
    #Initialize Homography Matrix
    H = np.zeros((3,3))

    #Setup the A Matrix 
    A = np.zeros((len(src_pts)*2,9))
    for i in range(len(src_pts)):
        A[i*2]=[0, 0, 0, -src_pts[i,0], -src_pts[i,1], -1, dest_pts[i,1]*src_pts[i,0], dest_pts[i,1]*src_pts[i,1], dest_pts[i,1]]
        A[i*2+1]=[src_pts[i,0], src_pts[i,1], 1, 0, 0, 0, -dest_pts[i,0]*src_pts[i,0], -dest_pts[i,0]*src_pts[i,1],-dest_pts[i,0]]
        
    #Do SVD Decomposition           
    U,D,V = np.linalg.svd(A)
    V_T = np.transpose(V) #Need to take transpose because rows of V are eigen vectors
    H_elements = V_T[:,-1] #Last column is the solution
    
    #Fill the Homography Matrix
    H[0] = H_elements[0:3] / H_elements[-1]
    H[1] = H_elements[3:6] / H_elements[-1]
    H[2] = H_elements[6:9] / H_elements[-1]
    return H

def getInliersCount(src_pts,dest_pts,H,delta):
    #Estimate of destination points
    dest_pts_estimate = np.zeros((dest_pts.shape),dtype='int')
    
    for src_pt in range(len(src_pts)):
        dest_pts_nonNorm = np.matmul(H,([src_pts[src_pt,0],src_pts[src_pt,1],1]))
        dest_pts_estimate[src_pt,0] = dest_pts_nonNorm[0]/dest_pts_nonNorm[-1]
        dest_pts_estimate[src_pt,1] = dest_pts_nonNorm[1]/dest_pts_nonNorm[-1]
    
    dest_pts_estimate_err = dest_pts_estimate - dest_pts
    dest_pts_estimate_err_sq = np.square(dest_pts_estimate_err)
    dist = np.sqrt(dest_pts_estimate_err_sq[:,0]+dest_pts_estimate_err_sq[:,1])

    Inliers=[1 for val in dist if (val < delta)]
    Inlier_count = len(Inliers)
    return Inlier_count

def RANSAC(combined_ip,delta,n,p,epsilon):
    #Seperate source and destination images XY coordinates
    src_xy = np.zeros((len(combined_ip),2))
    dest_xy = np.zeros((len(combined_ip),2))
    src_xy = combined_ip[:,0:2]
    dest_xy = combined_ip[:,2:]

    #Number of trials for determining homography
    N = int(math.log(1 - p)/math.log(1-(1-epsilon)**n))

    #Minimum value of inliner set considered acceptable
    M = int(len(src_xy) * (1-epsilon))

    #Initialize Homography Matrix
    H_trial = np.zeros((3,3))
    sol_list=[]

    #Loop over the total number of trials
    for trial in range(N):    
        #Randomly select n number of correspondences
        ip_index = np.random.randint(0,len(src_xy),n)
        src_pts_trial = src_xy[ip_index,:]
        dest_pts_trial = dest_xy[ip_index,:]
        
        #Calculate Homography by SVD for n selected correspondences
        H_trial = LinearLeastSquaresHomography(src_pts_trial,dest_pts_trial)

        #Count the number of Inliners
        InlierCount = getInliersCount(src_xy,dest_xy,H_trial,delta)
        H_vec = np.reshape(H_trial,9)
        
        if InlierCount > M:
            sol_list.append([H_vec,InlierCount])
        
    # Get a list of all possible homographies which satisfy threshold criterion
    all_solutions = np.array(sol_list)
    maxInlierIndx = np.argmax(all_solutions[:,-1])

    #Homograhy with the maximum inlier support
    H_ransac = np.reshape(all_solutions[maxInlierIndx,0],(3,3))
    return H_ransac

def displayImagewithInterestPointsandOutliers(img1,img2,corners,H,delta):
    #Get shape of the output image
    nrows = max(img1.shape[0], img2.shape[0])
    ncol = img1.shape[1]+img2.shape[1]
    
    #Initialize combined output image
    out_img = np.zeros((nrows,ncol,3))
    
    #Copy Image 1 to left half of the output image
    out_img[:img1.shape[0], :img1.shape[1]] = img1
    
    #Copy Image 2 to right half of the output image
    out_img[:img2.shape[0], img1.shape[1]:img1.shape[1]+img2.shape[1]] = img2

    #Seperate source and destination images XY coordinates
    src_pts = np.zeros((len(corners),2))
    dest_pts = np.zeros((len(corners),2))
    src_pts = corners[:,0:2]
    dest_pts = corners[:,2:]

    inliers_src_list = [] #list of inliers in source image
    inliers_dest_list = [] #list of inliers in source image
    for src_pt in range(len(src_pts)):
        dest_pt_estimate = np.matmul(H,[src_pts[src_pt,0],src_pts[src_pt,1],1])
        dest_pt_estimate = dest_pt_estimate/dest_pt_estimate[-1]
        diff = dest_pt_estimate[0:2] - dest_pts[src_pt,:]
        err_dist_dest_pt = np.sqrt(np.sum(diff**2))
        if  err_dist_dest_pt < delta: 
            inliers_src_list.append(src_pts[src_pt,:])
            inliers_dest_list.append(dest_pts[src_pt,:])
            cv2.circle(out_img,(int(src_pts[src_pt,0]),int(src_pts[src_pt,1])),2,(255,0,0),2)
            cv2.circle(out_img,(img1.shape[1]+int(dest_pts[src_pt,0]),int(dest_pts[src_pt,1])),2,(255,0,0),2)
            cv2.line(out_img,(int(src_pts[src_pt,0]),int(src_pts[src_pt,1])),(img1.shape[1]+int(dest_pts[src_pt,0]),int(dest_pts[src_pt,1])), (0,255,0))
        else :
            cv2.circle(out_img,(int(src_pts[src_pt,0]),int(src_pts[src_pt,1])),2,(0,0,255),2)
            cv2.circle(out_img,(img1.shape[1]+int(dest_pts[src_pt,0]),int(dest_pts[src_pt,1])),2,(0,0,255),2)
            cv2.line(out_img,(int(src_pts[src_pt,0]),int(src_pts[src_pt,1])),(img1.shape[1]+int(dest_pts[src_pt,0]),int(dest_pts[src_pt,1])), (0,0,255))

    return out_img, np.array(inliers_src_list), np.array(inliers_dest_list)

def BilinearInterpforPixelValue(img, pt):
    #Get coordinates of adjacent 4 points
    pt_imin1_jmin1 = img[(math.floor(pt[1])),(math.floor(pt[0]))]
    pt_iplus1_jmin1 = img[math.floor(pt[1]),math.floor(pt[0]+1)]
    pt_imin1_jplus1 = img[math.floor(pt[1]+1),math.floor(pt[0])]
    pt_iplus1_jplus1 =img[math.floor(pt[1]+1),math.floor(pt[0]+1)]

    #Calculate weights of points
    xdiff = pt[0] - math.floor(pt[0])
    ydiff = pt[1] - math.floor(pt[1])
    pt_imin1_jmin1_wt= pow(pow(xdiff,2) + pow(ydiff,2),-0.5)
    pt_iplus1_jmin1_wt = pow(pow(1-xdiff,2) + pow(ydiff,2),-0.5)
    pt_imin1_jplus1_wt = pow(pow(xdiff,2) + pow(1-ydiff,2),-0.5)
    pt_iplus1_jplus1_wt = pow(pow(1-xdiff,2) + pow(1-ydiff,2),-0.5)

    #Interpolated point
    result_num = pt_imin1_jmin1 * pt_imin1_jmin1_wt + pt_iplus1_jmin1 * pt_iplus1_jmin1_wt + pt_imin1_jplus1 * pt_imin1_jplus1_wt + pt_iplus1_jplus1 * pt_iplus1_jplus1_wt
    result_denom = pt_imin1_jmin1_wt + pt_iplus1_jmin1_wt + pt_imin1_jplus1_wt + pt_iplus1_jplus1_wt
    result = result_num / result_denom
    return result

def ImageExtent(img,H):
    img_corners = np.zeros((3,4))
    img_corners[:,0] = [0,0,1]
    img_corners[:,1] = [0,img.shape[1],1]
    img_corners[:,2] = [img.shape[0],0,1]
    img_corners[:,3] = [img.shape[0],img.shape[1],1]

    img_corners_range = np.matmul(H,img_corners)

    for i in range(img_corners_range.shape[1]):
        img_corners_range[:,i] = img_corners_range[:,i]/img_corners_range[-1,i]  

    return img_corners_range[0:2,:]

def getPanaromicImage(range_img,domain_img,H,offsetXY):
    H_inv = np.linalg.inv(H)
    for i in range(0,range_img.shape[0]): #Y-coordinate, row
        for j in range(0,range_img.shape[1]): #X-coordinate, col
                X_domain = np.array([j+offsetXY[0],i+offsetXY[1], 1])
                X_range = np.array(np.matmul(H_inv,X_domain))
                X_range = X_range/X_range[-1]
                
                if (X_range[0]>0 and X_range[1]>0 and X_range[0]<domain_img.shape[1]-1 and X_range[1]<domain_img.shape[0]-1):
                    range_img[i][j] = BilinearInterpforPixelValue(domain_img,X_range)
    return range_img

def getJacobian(src_pts,H):
    #Initialize Jacobian
    J = np.zeros((2*len(src_pts),9))

    #Calculate each row of the jacobian
    for src_pt in range(len(src_pts)):
        fp = np.matmul(H,([src_pts[src_pt,0],src_pts[src_pt,1],1]))
        J[src_pt*2]=[src_pts[src_pt,0]/fp[-1],src_pts[src_pt,1]/fp[-1],1/fp[-1],0,0,0,-src_pts[src_pt,0]*fp[0]/(fp[-1]**2),-src_pts[src_pt,1]*fp[0]/(fp[-1]**2),-1*fp[0]/(fp[-1]**2)]
        J[src_pt*2+1]=[0,0,0,src_pts[src_pt,0]/fp[-1],src_pts[src_pt,1]/fp[-1],1/fp[-1],-src_pts[src_pt,0]*fp[1]/(fp[-1]**2),-src_pts[src_pt,1]*fp[1]/(fp[-1]**2),-1*fp[1]/(fp[-1]**2)]
    return J

def applyHomography(src_pts,H):
    dest_pts = np.zeros((src_pts.shape))
    fp_list = []
    for src_pt in range(len(src_pts)):
        dest_pt = np.matmul(H,([src_pts[src_pt,0],src_pts[src_pt,1],1]))
        dest_pts[src_pt,0] = dest_pt[0]/dest_pt[-1]
        dest_pts[src_pt,1] = dest_pt[1]/dest_pt[-1]
        fp_list.append(dest_pts[src_pt,0])
        fp_list.append(dest_pts[src_pt,1])
    return np.array(fp_list)

def calCost(epsilon):
    return np.linalg.norm(epsilon)**2

def Levenberg_Marquardt_Method(src_pts,dest_pts_true,H_Linear):
    I = np.identity(9)
    tau = 0.5 #for initial value of mu
    Jf = getJacobian(src_pts,H_Linear)
    JfT_Jf = np.matmul(np.transpose(Jf),Jf)

    #Initial conditions
    mu_k = tau * np.max(np.diagonal(JfT_Jf))
    H_k = H_Linear

    X_true = []
    for dest_pt in range(len(dest_pts_true)):
        X_true.append(dest_pts_true[dest_pt,0])
        X_true.append(dest_pts_true[dest_pt,1])

    X_true = np.array(X_true)

    total_iter = 50
    niter = 0
    cost_LM = []
    while niter < total_iter:
        print("Iteration number = ", niter)
        print(".......")
        #Calculation of cost function C and error epsp_k at time step k
        fp_k = applyHomography(src_pts,H_k)
        epsp_k = X_true - fp_k
        Cp_k = calCost(epsp_k)

        #Calculation of deltaP
        Jf = getJacobian(src_pts,H_k)
        Jf_T = np.transpose(Jf)
        delta_p1 = np.linalg.inv(np.matmul(Jf_T,Jf) + mu_k*I)
        delta_p2 = np.matmul(delta_p1,Jf_T)
        delta_p = np.matmul(delta_p2,epsp_k)  
    
        #Calculation of cost function and error at time step k+1
        H_kp1 = H_k + np.reshape(delta_p,(3,3))
        fp_kp1 = applyHomography(src_pts,H_kp1)
        epsp_kp1 = X_true - fp_kp1
        Cp_kp1 = calCost(epsp_kp1)
        cost_LM.append(Cp_kp1)
        print("Cp_kp1 = ", Cp_kp1)

        #Calculation of quality measurement ratio rho
        rho_num = Cp_k - Cp_kp1
        rho_denom_term1 = np.matmul(np.matmul(delta_p,Jf_T),epsp_k)
        rho_denom_term2 = np.matmul(np.matmul(delta_p,mu_k*I),delta_p)
        rho_LM = rho_num / (rho_denom_term1 + rho_denom_term2)

        #Update damping coefficient
        mu_kp1 = mu_k * max(1/3, 1-(2*rho_LM-1)**3)

        #Reassign values
        mu_k = mu_kp1
        H_k = H_kp1
        niter = niter + 1

    #Plot cost function against number of iterations
    plt.figure()
    plt.scatter(range(total_iter), np.array(cost_LM), s=80, edgecolor="black", c="red", label="data")
    plt.xlabel("Iteration Number")
    plt.ylabel("Cost C(p)")
    plt.title("Variation of Cost with Iteration Number")
    plt.legend()
    plt.show()

    H_NonLinear = H_kp1
    return H_NonLinear

if __name__ == "__main__":

############################# SIFT DETECT INTEREST POINTS ############################   

    image1=cv2.imread("../HW5Pics/1.jpg")
    image2=cv2.imread("../HW5Pics/2.jpg")
    image3=cv2.imread("../HW5Pics/3.jpg")
    image4=cv2.imread("../HW5Pics/4.jpg")
    image5=cv2.imread("../HW5Pics/5.jpg")
  
    image1_gray=cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
    image2_gray=cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)
    image3_gray=cv2.cvtColor(image3,cv2.COLOR_BGR2GRAY)
    image4_gray=cv2.cvtColor(image4,cv2.COLOR_BGR2GRAY)
    image5_gray=cv2.cvtColor(image5,cv2.COLOR_BGR2GRAY)

    #Get interest points and descriptor for all images
    ip1, descp1 = SIFTDetectInterestPoints(image1_gray)
    ip2, descp2 = SIFTDetectInterestPoints(image2_gray)
    ip3, descp3 = SIFTDetectInterestPoints(image3_gray)
    ip4, descp4 = SIFTDetectInterestPoints(image4_gray)
    ip5, descp5 = SIFTDetectInterestPoints(image5_gray)

    #Establish correspondence using Brute Force Method
    ip_matches12 = BruteForceMatcher(descp1, descp2)
    ip_matches23 = BruteForceMatcher(descp2, descp3)
    ip_matches34 = BruteForceMatcher(descp3, descp4)
    ip_matches45 = BruteForceMatcher(descp4, descp5)
    
    #Get XY coordinates of corresponding Interest Points
    ip_matches12_XYCoord = getXYCoordinateInteresPoints(ip_matches12,ip1,ip2)
    ip_matches23_XYCoord = getXYCoordinateInteresPoints(ip_matches23,ip2,ip3)
    ip_matches34_XYCoord = getXYCoordinateInteresPoints(ip_matches34,ip3,ip4)
    ip_matches45_XYCoord = getXYCoordinateInteresPoints(ip_matches45,ip4,ip5)

    #Plot combined image with corresponding Interest Points
    out_image12 = displayImagewithInterestPoints(image1,image2,ip_matches12_XYCoord)
    out_image23 = displayImagewithInterestPoints(image2,image3,ip_matches23_XYCoord)
    out_image34 = displayImagewithInterestPoints(image3,image4,ip_matches34_XYCoord)
    out_image45 = displayImagewithInterestPoints(image4,image5,ip_matches45_XYCoord)
    #out_image=plotting(ip_matches_XYCoord,image1,image2)
    cv2.imwrite('ImagePair12.jpg',out_image12)
    cv2.imwrite('ImagePair23.jpg',out_image23)
    cv2.imwrite('ImagePair34.jpg',out_image34)
    cv2.imwrite('ImagePair45.jpg',out_image45)

 ############################# RANSAC TO REMOVE OUTLIERS ############################ 
    sigma_RSC = 1
    delta_RSC = 3 * sigma_RSC
    n_RSC = 6
    p_RSC = 0.999
    eps_RSC = 0.4

    H_Ransac12 = RANSAC(ip_matches12_XYCoord,delta_RSC,n_RSC,p_RSC,eps_RSC)
    out_image_with_outliers12, inliers_src12, inliers_dest12 = displayImagewithInterestPointsandOutliers(image1,image2,ip_matches12_XYCoord,H_Ransac12,delta_RSC)
    cv2.imwrite('ImagePair12_with_outliers.jpg',out_image_with_outliers12)

    H_Ransac23 = RANSAC(ip_matches23_XYCoord,delta_RSC,n_RSC,p_RSC,eps_RSC)
    out_image_with_outliers23, inliers_src23, inliers_dest23 = displayImagewithInterestPointsandOutliers(image2,image3,ip_matches23_XYCoord,H_Ransac23,delta_RSC)
    cv2.imwrite('ImagePair23_with_outliers.jpg',out_image_with_outliers23)

    H_Ransac34 = RANSAC(ip_matches34_XYCoord,delta_RSC,n_RSC,p_RSC,eps_RSC)
    out_image_with_outliers34, inliers_src34, inliers_dest34 = displayImagewithInterestPointsandOutliers(image3,image4,ip_matches34_XYCoord,H_Ransac34,delta_RSC)
    cv2.imwrite('ImagePair34_with_outliers.jpg',out_image_with_outliers34)

    H_Ransac45 = RANSAC(ip_matches45_XYCoord,delta_RSC,n_RSC,p_RSC,eps_RSC)
    out_image_with_outliers45, inliers_src45, inliers_dest45 = displayImagewithInterestPointsandOutliers(image4,image5,ip_matches45_XYCoord,H_Ransac45,delta_RSC)
    cv2.imwrite('ImagePair45_with_outliers.jpg',out_image_with_outliers45)

######################## HOMOGRAPHY REFINIEMENT WITH LEVENBERG-MARQUARDT NONLINEAR LEAST-SQUARE ############################ 
    H_LM12 = Levenberg_Marquardt_Method(inliers_src12,inliers_dest12,H_Ransac12)
    H_LM23 = Levenberg_Marquardt_Method(inliers_src23,inliers_dest23,H_Ransac23)
    H_LM34 = Levenberg_Marquardt_Method(inliers_src34,inliers_dest34,H_Ransac34)
    H_LM45 = Levenberg_Marquardt_Method(inliers_src45,inliers_dest45,H_Ransac45)

 ############################# IMAGE MOASICING WITHOUT NON-LINEAR ############################ 
    H_Ransac13 = np.matmul(H_Ransac12,H_Ransac23)
    H_Ransac13 = H_Ransac13/H_Ransac13[-1,-1]

    H_Ransac23 = H_Ransac23/H_Ransac23[-1,-1]

    H_Ransac43 = np.linalg.inv(H_Ransac34)
    H_Ransac43 = H_Ransac43/H_Ransac43[-1,-1]

    H_Ransac35 = np.matmul(H_Ransac34,H_Ransac45)
    H_Ransac53 = np.linalg.inv(H_Ransac35)
    H_Ransac53 = H_Ransac53/H_Ransac53[-1,-1]
    
    H_Ransac33 = np.identity(3)

    #Get Size of the Final Panarmoic Image
    corners13 = ImageExtent(image1,H_Ransac13)
    corners23 = ImageExtent(image2,H_Ransac23)
    corners33 = ImageExtent(image3,H_Ransac33)
    corners43 = ImageExtent(image4,H_Ransac43)
    corners53 = ImageExtent(image5,H_Ransac53)
    min_xy_coord = np.amin(np.amin([corners13,corners23,corners33,corners43,corners53],2),0)
    max_xy_coord = np.amax(np.amax([corners13,corners23,corners33,corners43,corners53],2),0)
    final_img_dim = max_xy_coord - min_xy_coord
    pan_img = np.zeros((int(final_img_dim[1]),int(final_img_dim[0]),3)) #Final size of panaromic image

    #Generate Final Panaromic Image
    print("Generating Panaromic Image without LM refinement ....")
    pan_img = getPanaromicImage(pan_img,image1,H_Ransac13,min_xy_coord)
    pan_img = getPanaromicImage(pan_img,image2,H_Ransac23,min_xy_coord)
    pan_img = getPanaromicImage(pan_img,image3,H_Ransac33,min_xy_coord)
    pan_img = getPanaromicImage(pan_img,image4,H_Ransac43,min_xy_coord)
    pan_img = getPanaromicImage(pan_img,image5,H_Ransac53,min_xy_coord)
    cv2.imwrite("PanaromicImageWithoutLM.jpg",pan_img)
    print("Image without LM refinement written")
 ################################################################### 

 ############################# IMAGE MOASICING WITH NON-LINEAR ############################ 
    H_LM13 = np.matmul(H_LM12,H_LM23)
    H_LM13 = H_LM13/H_LM13[-1,-1]

    H_LM23 = H_LM23/H_LM23[-1,-1]

    H_LM43 = np.linalg.inv(H_LM34)
    H_LM43 = H_LM43/H_LM43[-1,-1]

    H_LM35 = np.matmul(H_LM34,H_LM45)
    H_LM53 = np.linalg.inv(H_LM35)
    H_LM53 = H_LM53/H_LM53[-1,-1]
    
    H_LM33 = np.identity(3)

    #Get Size of the Final Panarmoic Image
    corners13_LM = ImageExtent(image1,H_LM13)
    corners23_LM = ImageExtent(image2,H_LM23)
    corners33_LM = ImageExtent(image3,H_LM33)
    corners43_LM = ImageExtent(image4,H_LM43)
    corners53_LM = ImageExtent(image5,H_LM53)
    min_xy_coord_LM = np.amin(np.amin([corners13_LM,corners23_LM,corners33_LM,corners43_LM,corners53_LM],2),0)
    max_xy_coord_LM = np.amax(np.amax([corners13_LM,corners23_LM,corners33_LM,corners43_LM,corners53_LM],2),0)
    final_img_dim_LM = max_xy_coord_LM - min_xy_coord_LM
    pan_img_LM = np.zeros((int(final_img_dim_LM[1]),int(final_img_dim_LM[0]),3)) #Final size of panaromic image

    print("Generating Panaromic Image with LM refinement ....")
    #Generate Final Panaromic Image
    pan_img_LM = getPanaromicImage(pan_img_LM,image1,H_LM13,min_xy_coord_LM)
    pan_img_LM = getPanaromicImage(pan_img_LM,image2,H_LM23,min_xy_coord_LM)
    pan_img_LM = getPanaromicImage(pan_img_LM,image3,H_LM33,min_xy_coord_LM)
    pan_img_LM = getPanaromicImage(pan_img_LM,image4,H_LM43,min_xy_coord_LM)
    pan_img_LM = getPanaromicImage(pan_img_LM,image5,H_LM53,min_xy_coord_LM)
    cv2.imwrite("PanaromicImageWithLM.jpg",pan_img_LM)
    print("Image with LM refinement written")

 ###################################################################   
    print("CODE RUN SUCCESS!!!!")
