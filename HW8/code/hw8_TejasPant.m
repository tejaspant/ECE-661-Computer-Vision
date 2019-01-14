%%ECE 661 Computer Vision HW 8
%Tejas Pant
%5th November 2018
%Camera Calibration using Zhang's Algorithm
function CameraCalibration()
fclose all; clear all; close all; clc;
%Input Section
pattern_size = 25; %mm
nptsx = 8; %Number of Corner Points in Calibration Pattern in the X direction
nptsy = 10; %Number of Corner Points in Calibration Pattern in the Y direction
x_world = zeros(80,2); 
radDistort = 0; % Flag to indicate if radial distortion is accounted or not
reprojection = 1; % Reproject corners

dataSet = 'Dataset2'; %Dataset used for Camera Calibration

if dataSet == 'Dataset1'
    nImagesSet = 38; %Somehow does not work when I use all 40 images
    dirLocSet = '..\Files\Dataset1\';
    hough_rho_resolution = 0.6;
    canny_detector_thresh = 0.8;
    fixedImageNum = 26; %Fixed Image Number in Dataset
    ImageNumForReproject = [13, 21, 23]; %Image Numbers used for Reprojection
else
    nImagesSet = 20;
    dirLocSet = '..\Files\Dataset2_Tejas';
    hough_rho_resolution = 0.7;
    canny_detector_thresh = 0.7;
    fixedImageNum = 1; %Fixed Image Number in Dataset
    ImageNumForReproject = [4, 8, 13]; %Image Numbers used for Reprojection
end

%Generate World Coordinates
%Size of black pattern in calibration pattern = 25mm x 25 mm
for i = 1:nptsx
    for j = 1:nptsy
        x_world((i-1)* nptsy + j,:)=[(i-1)*pattern_size (j-1)*pattern_size];
    end
end

HomoWorldCoordPixelCoordAll = []; %Stores homography between world coordinates and pixel coordinates for all images
V_zhang = []; %V matrix used in zhang's algrotihm
CornersPixelCoordAll = []; %Corner Pixel Cooridnates for all Images
for iImg = 1:nImagesSet
    if iImg == 10 & dataSet == 'Dataset1'
		%This Image seems to be problematic. Hence special treatment
        hough_rho_resolution = 0.5;
    end
    imageName = strcat([dirLocSet, '\Pic_' num2str(iImg), '.jpg']);
    [pixel_coord] = detectCorners(imageName,iImg,hough_rho_resolution,canny_detector_thresh);
    if length(pixel_coord) ~= nptsx * nptsy
        fprintf('Number of corners not detected correctly for image number %d \n',iImg);
    end
    CornersPixelCoordAll{iImg} = pixel_coord;
    
    %Use Linear-Least Square Homography Estimation. min||Ah||, ||h||=1
    %pixel coordinates = H * world Cooridnates
    A = setupA(x_world(:,1),x_world(:,2),double(pixel_coord(:,1)),double(pixel_coord(:,2)));
    [U,S,V] = svd(A);
    H = [V(1:3,9)'; V(4:6,9)'; V(7:9,9)'];
    HomoWorldCoordPixelCoordAll{iImg} = H;
    
    %Calculate Zhang Algorithm Vij matrix
    [V12, V11, V22] = calcVZhangFromHomography(H);
    V_zhang = [V_zhang; V12;(V11-V22)];
end

%SVD of V in Zhang's algorithm to get b
[U,D,V] = svd(V_zhang);
b = V(:,6); 

%Get intrinsic parameters from b
[x0, lambda, alphaX, alphaY, s, y0] = getIntrinsicParamFromb(b);
K = [alphaX s x0; 0 alphaY y0; 0 0 1];

%vector p used in cost function optimization using LM
vec_p = zeros(1,5+6*nImagesSet);
vec_p(1:5) = [alphaX s x0 alphaY y0];

if(radDistort)
	vec_p = zeros(1,7+6*nImagesSet);
	vec_p(1:5) = [alphaX s x0 alphaY y0];
	vec_p(6:7) = [0 0];
	count = 7;
else
	vec_p = zeros(1,5+6*nImagesSet);
	vec_p(1:5) = [alphaX s x0 alphaY y0];
	count = 5;
end

ydata = [];
KInv = pinv(K);
R_LinLeast = [];
t_LinLeast = [];

%Calculation of intrinsic parameters based on Linear Least Square Estimation of Homography
for k = 1:nImagesSet
     H = HomoWorldCoordPixelCoordAll{k};
     h1 = H(:,1); h2 = H(:,2); h3 = H(:,3);
     t = KInv * h3;
     zeta = 1 / norm(KInv * h1);
     if(t(3)<0)
     zeta = -zeta;
     end
     r1 = zeta * KInv * h1;
     r2 = zeta * KInv * h2;
     r3 = cross(r1,r2);
     R = [r1 r2 r3];
     t = zeta * t;
     [U,D,V] = svd(R);
     R = U * V';
     R_LinLeast{k} = R;
     t_LinLeast{k} = t;
	 
     % Rodriguez Representation of Rotation Matrix
     phi = acos((trace(R)-1)/2);
     w = phi/(2*sin(phi))*([R(3,2)-R(2,3) R(1,3)-R(3,1) R(2,1)-R(1,2)])';
     vec_p(count+1:count+3) = w;
     vec_p(count+4:count+6) = t;
     count = count + 6;
     y = CornersPixelCoordAll{k};
     y = y';
     ydata = [ydata y(:)'];
end
x = x_world';
xdata = x(:)';

% LM algorithm is carried out for refinement
options = optimoptions('lsqcurvefit','Algorithm','levenberg-marquardt');
vec_p_LM = lsqnonlin(@costFunctionLM,vec_p,[],[],options,xdata,ydata,radDistort,nImagesSet);

% Finding the intrinsic calibration matrix
alphaX = vec_p_LM(1);
s = vec_p_LM(2);
x0 = vec_p_LM(3);
alphaY = vec_p_LM(4);
y0 = vec_p_LM(5);
K1 = [alphaX s x0; 0 alphaY y0; 0 0 1];

if(radDistort)
	k1 = vec_p_LM(6);
	k2 = vec_p_LM(7);
	count = 7;
else
	count = 5;
end

R_LM = [];
t_LM = [];
%Get Extrinsic Parameters from vector p
for k = 1:nImagesSet
     w = vec_p_LM(count+1:count+3);
     t_LM{k} = vec_p_LM(count+4:count+6)';
     count = count + 6;
     wx = [0 -w(3) w(2); w(3) 0 -w(1); -w(2) w(1) 0];
     phi = norm(w);
     R_LM{k} = eye(3)+ sin(phi)/phi * wx + (1-cos(phi))/phi*wx^2;
end

%Reprojection of corners
if reprojection == 1
    for iImg = 1:length(ImageNumForReproject)
        [mean_err_beforeLM(iImg), var_err_beforeLM(iImg)] = reprojectCorners(dirLocSet,fixedImageNum,R_LinLeast,t_LinLeast,K,CornersPixelCoordAll,ImageNumForReproject(iImg));
        [mean_err_afterLM(iImg), var_err_afterLM(iImg)] = reprojectCorners(dirLocSet,fixedImageNum,R_LM,t_LM,K1,CornersPixelCoordAll,ImageNumForReproject(iImg));
    end
end
end
%%
%Detect corners using Canny edge detector and Hough Transform
function [corner] = detectCorners(imgName,imgNum,rhoResol,cannyThresh)
    
    %Read image and convert to grayscal form
    img = imread(imgName);
    img_gray = rgb2gray(img);
    
    %Detect Edges using Cann edge detector
    img_edge = edge(img_gray,'canny',cannyThresh); 
    
    %Show pattern with edges
    figure(100)
    imshow(img_edge)
    
    %Hough Transform
    [H, theta, rho] = hough(img_edge,'RhoResolution',rhoResol); %
    
    %Identify peaks in the hough transform
    %Total of 18 horizontal and vertical lines. Need to detect 18 peaks
    P = houghpeaks(H,18,'Threshold',15); 
    
    %Extract lines based on Hough transform
    lines = houghlines(img_edge,theta,rho,P,'FillGap',150,'MinLength',70);
    
    %for our and provided dataset 150 and 70
    %Store parameters of the line: slope(m), y-intercept(c)
    %Line: y = mx + c
    
    lineParams = zeros(length(lines),2); %slope, y-intersect,
    %Initializing the horizontal and vertical
    hor = []; ver = [];
    
    %Identify horizontal and vertical lines
    for k = 1:length(lines)
     xy = [lines(k).point1; lines(k).point2];
     
     %Get Slope of the detected line
     lineParams(k,1) = (xy(1,2)-xy(2,2))/(xy(1,1)-xy(2,1));
     
     % plot_line(lines,k,size(img_edge));
     if(abs(lineParams(k,1))>1)
     ver = [ver k];
     else
     hor = [hor k];
     end
     if(abs(lineParams(k,1)) == inf)
     lineParams(k,2) = inf;
     else
     lineParams(k,2) = xy(1,2) - lineParams(k,1)*xy(1,1);
     end
    end
    %Initializing the list for the corners
    corner = [];
    for i = 1:length(lines)
     n_c{i} = [];
    end

    %This is used to get rid of the extra lines
    lines_hor = lines(hor);
    ehor = zeros(1,length(hor));
    for i= 1:length(lines_hor)
     for j = i+1:length(lines_hor)
     [pt] = TwoLinesInterSectionPoint(lines_hor(i), lines_hor(j));
     if(pt(1)>1 && pt(1)<size(img,2) && pt(2)>1 && pt(2)<size(img,1))
     ehor(i) =ehor(i)+ 1;
     ehor(j) = ehor(j)+1;
     end
     end
    end
    lines_ver = lines(ver);
    ever = zeros(1,length(ver));
    for i= 1:length(lines_ver)
     for j = i+1:length(lines_ver)
     [pt]= TwoLinesInterSectionPoint(lines_ver(i), lines_ver(j));
     if(pt(1)>1 && pt(1)<size(img,2) && pt(2)>1 && pt(2)<size(img,1))
     ever(i) = ever(i) +1;
     ever(j) = ever(j) +1;
     end
     end
    end
    
    [ever ind1] = sort(ever,'ascend');
    [ever ind2] = sort(ehor,'ascend');
    if (length(hor) ~= 10)
        fprintf('Issue with image number %d',imgNum);
    end
    lines = lines([hor(ind2(1:10)) ver(ind1(1:8))]);

    %Plotting the detected lines
    figure
    imshow(img_gray)
    for k = 1:length(lines)
     ptxy = [lines(k).point1; lines(k).point2];
     %find the equation of the line y = mx + b
     %find slope m
     lineParams(k,1) = (ptxy(1,2)-ptxy(2,2))/(ptxy(1,1)-ptxy(2,1));
     if(abs(lineParams(k,1)) == inf)
     lineParams(k,2) = inf;
     hold on
     y = 1:size(img,1);
     x = ptxy(1,1)*ones(1,length(y));
     plot(x,y,'Color','green')
     else
     lineParams(k,2) = ptxy(1,2) - lineParams(k,1)*ptxy(1,1);
     f = @(x) lineParams(k,1)*x + lineParams(k,2);
     x = 1:size(img,2);
     y = uint64(f(x));
     hold on
     plot(x,y,'Color','green');
     end
    end
    
    %Plotting the corners 
    for i= 1:length(lines)
         for j = i+1:length(lines)
         [pt] = TwoLinesInterSectionPoint(lines(i), lines(j));
             if(pt(1)>1 && pt(1)<size(img,2) && pt(2)>1 && pt(2)<size(img,1))
             corner = [corner; pt ];
             n_c{i} = [n_c{i} size(corner,1)];
             n_c{j} = [n_c{j} size(corner,1)];
             end
         end
    end

    hor = [];
    ver = [];
    for i = 1:length(lines)
         if(length(n_c{i}) == 8)
             hor = [hor i];
         else
             ver = [ver i];
         end
    end
    xs = zeros(length(ver),1); 
    for i = 1:length(ver)
        ind = n_c{ver(i)};
        xs(i) = min(corner(ind,1)); 
    end
    [d ind] = sort(xs,'ascend'); 
    ver = ver(ind); 
    labels = zeros(80,1);
    count = 0;
    ys = zeros(10,1); 
    for i = 1:length(ver)
         ind = n_c{ver(i)}; 
         ys = corner(ind,2);
         [d sind] = sort(ys,'ascend');
         for j = 1:length(sind) 
             count =count + 1;
             labels(count) = ind(sind(j));
         end
    end
    corner = corner(labels,:);

    for i = 1:length(labels)
         hold on
         text(corner(i,1),corner(i,2),int2str(i),'Color','r');
    end  
end
%%
%Calculates intersection point of two lines using HC Representation
function [intersect_pt] = TwoLinesInterSectionPoint(line1, line2)
%Convert end points of line to HC representation
ptA_line1 = [line1.point1 1];
ptB_line1 = [line1.point2 1];

%HC representation of first line
lineA = cross(ptA_line1, ptB_line1);

%Convert end points of line to HC representation
ptA_line2 = [line2.point1 1];
ptB_line2 = [line2.point2 1];

%HC representation of second line
lineB = cross(ptA_line2, ptB_line2); 

%HC representation of intersection point of lineA and Line B
intersect_pt = cross(lineA, lineB);

%Get XY coordinates
intersect_pt = double([intersect_pt(1)/intersect_pt(3) intersect_pt(2)/intersect_pt(3)]);
end
%%
%Setup A for calculation of homography using linear-least square minimization
function [A] = setupA(xdomain,ydomain, xrange, yrange)
A = zeros(2*length(xdomain),9);
for i = 1:length(xdomain)
     A(2*(i-1)+1,:) = [0 0 0 -xdomain(i) -ydomain(i) -1 xdomain(i) * yrange(i) ydomain(i) * yrange(i) yrange(i)];
     A(2*i,:) = [xdomain(i) ydomain(i) 1 0 0 0 -xdomain(i)*xrange(i) -ydomain(i)*xrange(i) -xrange(i)];
end
end
%%
%Calculates V12, V11 and V22 used in matrix V in Zhang's algorithm
function [v12, v11, v22] = calcVZhangFromHomography(H)
    i=1; 
    j=2;
    v12 = [H(1,i)*H(1,j), H(1,i)*H(2,j)+H(2,i)*H(1,j), H(2,i)*H(2,j), H(3,i)*H(1,j)+H(1,i)*H(3,j) ,H(3,i)*H(2,j)+H(2,i)*H(3,j),H(3,i)*H(3,j)];
 
    i=1; 
    j=1;
    v11 = [H(1,i)*H(1,j), H(1,i)*H(2,j)+H(2,i)*H(1,j), H(2,i)*H(2,j), H(3,i)*H(1,j)+H(1,i)*H(3,j) ,H(3,i)*H(2,j)+H(2,i)*H(3,j),H(3,i)*H(3,j)];
    
    i=2;
    j=2;
    v22 = [H(1,i)*H(1,j), H(1,i)*H(2,j)+H(2,i)*H(1,j), H(2,i)*H(2,j), H(3,i)*H(1,j)+H(1,i)*H(3,j) ,H(3,i)*H(2,j)+H(2,i)*H(3,j),H(3,i)*H(3,j)];
end
 %%
 %Calculates Intrinsic Paramters from Image of Conic in Infinity Plane
 function [x0, lambda, alphaX, alphaY, s, y0] = getIntrinsicParamFromb(b)
    x0 = (b(2)* b(4)- b(1)* b(5))/(b(1) * b(3) - b(2)^2);
    lambda = b(6)-(b(4)^2 + x0 *(b(2) * b(4) - b(1) * b(5)))/b(1);
    alphaX = sqrt(lambda / b(1));
    alphaY = sqrt(lambda * b(1)/(b(1)*b(3)-b(2)^2));
    s = -b(2) * alphaX^2 * alphaY/lambda;
    y0 = s*x0/alphaY - (b(4)*alphaX^2/lambda);
 end
%%
%Cost Function for LM method
function error = costFunctionLM(p,worldCoord,pixelCoord,rad_dist,nimg)
ax = p(1);
s = p(2);
x0 = p(3);
ay = p(4);
y0 = p(5);
K = [ax s x0; 0 ay y0; 0 0 1]; % The intrinsic calibration matrix
if(rad_dist == 1)
	k1 = p(6);
	k2 = p(7); % These are the parameters of radial distortion
	K1 = [ax 0 x0; 0 ay y0; 0 0 1];
	count = 7;
else
	count = 5;
end

projPixelCoord = zeros(1,nimg*160);
n1 = 1;
for k = 1:nimg
     % Converting to the R,t using Rodriguez formula
     w = p(count+1:count+3);
     t = p(count+4:count+6)';
     count = count + 6;
     wx = [0 -w(3) w(2); w(3) 0 -w(1); -w(2) w(1) 0];
     phi = norm(w);
     R = eye(3)+ sin(phi)/phi * wx + (1-cos(phi))/phi * wx^2;
     n2 = 1;
     for i = 1:80
         % Projection for all the corner points onto the fixed image.  
         x = K * [R t] * [worldCoord(n2:n2+1) 0 1]';
         projPixelCoord(n1:n1+1) = [x(1)/x(3) x(2)/x(3)];
         if(rad_dist == 1)
            xp = [projPixelCoord(n1:n1+1) 1];
            xw = pinv(K1)*xp';
            r2 = xw(1)^2 + xw(2)^2;
            xp1 = xw(1) + xw(1)*(k1*r2+k2*r2^2);
            xp2 = xw(2) + xw(2)*(k1*r2+k2*r2^2);
            x = K1*[xp1 xp2 1]';
            projPixelCoord(n1:n1+1) = [x(1)/x(3) x(2)/x(3)];
         end
         n1 = n1+2;
         n2 = n2+2;
     end
end
error = pixelCoord - projPixelCoord;
end
%%
%Reprojects corners from an image on to the fixed image
function [mean_err, var_err] = reprojectCorners(dirLoc,fixedImgNum,R,t,K,pixelCoordCorners,imgNum)

filename = strcat([dirLoc, '/Pic_' num2str(fixedImgNum), '.jpg']);
img = rgb2gray(imread(filename));

%Homography of the fixed image
P_fixedImg = K * [R{fixedImgNum}(:,1:2) t{fixedImgNum}];
xtrue = pixelCoordCorners{fixedImgNum};

%Homography of selected imaged
P = K * [R{imgNum}(:,1:2) t{imgNum}];%This is the Homography for the projected image
pixelCoordImg = pixelCoordCorners{imgNum};
pixelCoordImg = [pixelCoordImg ones(size(pixelCoordImg,1),1)];

worldCoordImg = pinv(P) * pixelCoordImg';
xProjected = (P_fixedImg * worldCoordImg)';

figure
imshow(img)

for i = 1:80
	xProjected(i,:) = xProjected(i,:) / xProjected(i,3);
	hold on
	plot(uint64(xtrue(i,1)),uint64(xtrue(i,2)),'g.','MarkerSize',12);
	plot(uint64(xProjected(i,1)),uint64(xProjected(i,2)),'r.','MarkerSize',12);
end
xProjected = xProjected(:,1:2);
hold off

%Calculating moments of the error
mean_err = mean(abs(xtrue(:) - xProjected(:))); 
var_err = var(abs(xtrue(:) - xProjected(:))); 
end