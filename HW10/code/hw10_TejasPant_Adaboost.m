%Adaboost Classifier for Car Detection
%Tejas Pant
%29th Nov 2019
function AdaBoostCarDetection()
clear all; close all; clc; fclose all;
%Select which stage
GenerateFeatures = 0; %Generate Haar Features for Training and Test Set
TrainingStage = 0; %Training stage, build cascade of strong classifiers
TestingStage = 1; %Test stage, prediction of test set using cascade

%Maximum Number of strong classifiers in cascade
N_StrongClassifiers = 10;

%Number of iterations for getting a strong classifier
N_iterations = 100;

thresh_TruePos = 1; %true positive rate
thresh_FalsPos = 0.5; %threshold for false positive rate

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%FEATURE GENERATION STAGE %%%%%%%%%%%%%%%%%%%%%%%%%%%
if GenerateFeatures == 1
    trainPath_pos = '..\ECE661_2018_hw10_DB2\train\positive\'; %Positive Cars
    trainPath_neg = '..\ECE661_2018_hw10_DB2\train\negative\'; %Negative Cars
    testPath_pos = '..\ECE661_2018_hw10_DB2\test\positive\'; %Positive Cars
    testPath_neg = '..\ECE661_2018_hw10_DB2\test\negative\'; %Negative Cars
    
    %Features for training set
    trainFeatures_pos = genAdaboostFeatures(trainPath_pos);
    save('trainFeatures_pos.mat','trainFeatures_pos','-v7.3');
    trainFeatures_neg = genAdaboostFeatures(trainPath_neg);
    save('trainFeatures_neg.mat','trainFeatures_neg','-v7.3');
    
    %Features for test set
    testFeatures_pos = genAdaboostFeatures(testPath_pos);
    save('testFeatures_pos.mat','testFeatures_pos','-v7.3');
    testFeatures_neg = genAdaboostFeatures(testPath_neg);
    save('testFeatures_neg.mat','testFeatures_neg','-v7.3');   
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%TRAINING STAGE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if TrainingStage == 1
    load('trainFeatures_pos.mat');
    load('trainFeatures_neg.mat');

    trainFeatures_pos = feature_train_p;
    trainFeatures_neg = feature_train_n;

    nPositive = size(trainFeatures_pos,2);
    nNegative = size(trainFeatures_neg,2);
    new_clf_idx = 1:1:nPositive + nNegative;
    
    %build strong classsifier comprising of number of weak classifiers
    for iStrClf = 1:N_StrongClassifiers
        Strong_Clf = buildStrongClassifier(trainFeatures_pos,trainFeatures_neg,new_clf_idx,thresh_TruePos,thresh_FalsPos,N_iterations);

        %list of indices in the training set after removing TN
        new_clf_idx = Strong_Clf.updatedIdx;
        Strong_Clf_train(iStrClf) = Strong_Clf;
        
        disp(['Cascade Stage ' num2str(iStrClf) ]);
        disp(['Number of Iterations = ', num2str(Strong_Clf.numIterations)]);
        disp(['Number of False Positives = ', num2str(length(new_clf_idx) - nPositive)]);
        disp(['False Positive Rate = ', num2str(Strong_Clf.FPRate)]);
        FPRateTrain(iStrClf) = Strong_Clf.FPRate;
        
        %Stopping criterion if FP rate = 0.0
        if (length(new_clf_idx) == nPositive)
            break;
        end

    end
    figure
    hold on 
    plot(1:1:length(FPRateTrain),FPRateTrain,'ro-','LineWidth',2, 'markersize', 10)
    hold off
    legend('False Positive Rate');
    xlabel('Cascade Stage','FontSize',20); 
    ylabel('Rate', 'FontSize',20);
    ylim([0 1]);
    box on
    set(gca,'LineWidth',1.5,'FontSize',20)
    save('Training_Stage.mat','Strong_Clf_train','-v7.3');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%TESTING STAGE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if TestingStage == 1
    %Load training set and trained cascade of strong classifiers
    load('testFeatures_pos.mat');
    load('testFeatures_neg.mat');
    testFeatures_pos = feature_test_p;
    testFeatures_neg = feature_test_n;
    nPos_test = size(testFeatures_pos,2);
    nNeg_test = size(testFeatures_neg,2);
    load('Training_Stage.mat');
    
    numCascadeStages = length(Strong_Clf_train); %number of cascade stages
    numFN_test = 0; %number of false negatives in test set
    numTN_test = 0; %number of true negatives in test set
    FPRate = zeros(numCascadeStages,1); %False positive rate at each stage
    FNRate = zeros(numCascadeStages,1); %False negative rate at each stage
    for istage = 1:numCascadeStages
        %Exract weak classifier information for each cascade stage
        Strong_Clf_Stage = Strong_Clf_train(istage);
        prediction_stage = predictionTestSet(testFeatures_pos,testFeatures_neg,Strong_Clf_Stage);
        nPos_stage = prediction_stage.nPos; %number of positive samples
        nNeg_stage = prediction_stage.nNeg; %number of negative samples
        Str_Clf_predict_stage = prediction_stage.H; %classification prediction
          
        %Calculate FN rate and FP rate
        nFalseNegative = length(find(Str_Clf_predict_stage(1:nPos_stage) == 0));
        numFN_test = numFN_test + nFalseNegative;
        FNRate(istage) = numFN_test / nPos_test;
        nTrueNegative = length(find(Str_Clf_predict_stage(nPos_stage+1:end) == 0));
        numTN_test = numTN_test + nTrueNegative;
        FPRate(istage) = (nNeg_test - numTN_test) / nNeg_test;
        
        %Throw away false negatives in positive set and true negatives in negative set
        
        pos_nextStage_idx = find(Str_Clf_predict_stage(1:nPos_stage) == 1);
        testFeatures_pos = testFeatures_pos(:,pos_nextStage_idx);
        neg_nextStage_idx = find(Str_Clf_predict_stage(nPos_stage+1:end) == 1);
        testFeatures_neg = testFeatures_neg(:,neg_nextStage_idx);
    end
    
    figure
    hold on 
    plot(1:1:numCascadeStages,FNRate,'ko-','LineWidth',2, 'markersize', 10)
    plot(1:1:numCascadeStages,FPRate,'ro-','LineWidth',2, 'markersize', 10)
    hold off
    legend('False Negative Rate','False Positive Rate');
    xlabel('Cascade Stage','FontSize',20); 
    ylabel('Rate', 'FontSize',20);
    ylim([0 1]);
    box on
    set(gca,'LineWidth',1.5,'FontSize',20)
    
end
end
%%
%Generate Haar Type features
function [img_features] = genAdaboostFeatures(path)
    imgs = dir([path,'*.png']);
    img = imread([path imgs(1).name]);
    [img_ht,img_wd,~] = size(img); % assume all training images have the same sizes
    nimgs = length(imgs); % number of images
    
    %Need to change this to get more number of features
    horFeat_size = 2:2:img_wd;
    vertFeat_size = 2:2:img_ht;
    
    %Total number of features. Need to change this
    nfeat = 11900;
    img_features = zeros(nfeat,nimgs);
    for i = 1:nimgs
        img = imread([path imgs(i).name]);
        img_gray = double(rgb2gray(img));
        intg_img = integralImage(img_gray);
        %Initialize feature vector for a single image
        feature_img = [];
        
        %Generate Type 1 features, derivatives in X direction Haar operator 
        % i,j      ----------------- i, j + wd/2 ----------------- i, j + wd
        %  |                          |                               |
        %  |          Rect_l          |              Rect_r           |
        %  |                          |                               |
        % i+1,j ----------------  i+1, j + wd/2  ----------------- i+1, j + wd
        for j  = 1:length(horFeat_size)
            width_ = horFeat_size(j);
            for k = 1:img_ht
                for l = 1:(img_wd - width_ + 1)
                    rect_l = [k l;k (l + width_/2);(k + 1) l;(k + 1) (l+width_/2)]; % 1,2,3,4 corners
                    rect_r = [k (l+width_/2);k (l+width_);k+1 (l+width_/2);k+1 l+width_];
                    
                    %Calculate sum of pixels in left and right rectangle using integral images
                    rect_l_sum = sumPixelsRect(intg_img, rect_l);
                    rect_r_sum = sumPixelsRect(intg_img, rect_r);
                    feature_img = [feature_img; (rect_r_sum - rect_l_sum)];
                end
            end
        end
        
        %Generate Type 2 features, derivatives in Y direction Haar operator
        % i,j      ----------------- i, j+2
        %  |                          |
        %  |          Rect_u          |
        %  |                          |
        % i+ht/2,j ----------------  i+ht/2, j+2
        %  |                          |
        %  |          Rect_b          |
        %  |                          |
        % i+ht,j  ----------------  i+ht, j+2
        for j  = 1:length(vertFeat_size)
            height_ = vertFeat_size(j);
            for k = 1:img_ht - height_ + 1
                for l = 1:img_wd - 1
                    rect_u = [k l;k l+2;k+height_/2 l;k+height_/2 l+2]; % 1,2,3,4 corners
                    rect_b = [k+height_/2 l;k+height_/2 l+2;k+height_ l;k+height_ l+2];
                    
                    %Calculate sum of pixels in left and right rectangle using integral images
                    rect_u_sum = sumPixelsRect(intg_img, rect_u);
                    rect_b_sum = sumPixelsRect(intg_img, rect_b);
                    feature_img = [feature_img; (rect_u_sum - rect_b_sum)];
                end
            end
        end
        img_features(:,i) = feature_img;
    end
end
%%
%Calculate sum of pixels within a rectangle using integral image representation
function sum_PixVal = sumPixelsRect(integ_img, rect_corners)
    cor1 = integ_img(rect_corners(1,1), rect_corners(1,2));
    cor2 = integ_img(rect_corners(2,1), rect_corners(2,2));
    cor3 = integ_img(rect_corners(3,1), rect_corners(3,2));
    cor4 = integ_img(rect_corners(4,1), rect_corners(4,2));
    sum_PixVal = cor4 + cor1 - cor2 - cor3;
end 
%%
%Determine the final strong classifier 
function [strongClass] = buildStrongClassifier(feat_pos,feat_neg,img_idx,thresh_TP,thresh_FP,totalIter)
nPos = size(feat_pos,2);
nImg_total = length(img_idx);
nNeg = nImg_total - nPos;

%Intialize quantities to store for weak classifiers
h = zeros(4,totalIter); %Store parameters of wak classifier
h_res = zeros(nImg_total,totalIter); %Classification result for a classifier
alpha = zeros(totalIter);

%Strong classifier
strongClass_result = zeros(nImg_total,1); %classification result for strong classifier
strongClass_TP = zeros(totalIter,1); %True positive accuracy for strong classifier
strongClass_FP = zeros(totalIter,1); %False positive accuracy for strong classifier

%Combined features of positive and negative 
feat_comb = [feat_pos,feat_neg];

%Dataset of images after removing TN detected
feat_update = feat_comb(:,img_idx);
    
%Allocate memory to weights and labels
weight = zeros(nImg_total,1);
label = zeros(nImg_total,1);
    
%Initialize weights
weight(1:nPos) = 1./2/nPos;
weight(nPos+1:end) = 1./2/nNeg;
    
%True labels for the data
label(1:nPos) = 1;
for iter = 1:totalIter %total number of iterations or basically number of weak classifiers
    weight = weight./sum(weight);
    %Get best weak classifier
    [WC_e, WC_p, WC_f, WC_res, WC_theta] = buildWeakClassifier(feat_update, nPos, nNeg, weight, label);
    h(1,iter) = WC_f; %feature number
    h(2,iter) = WC_theta; %value of theta
    h(3,iter) = WC_p; %value of polarity p
    h_res(:,iter) = WC_res; %classification result for all images using weak classifier
    eps_t = WC_e; %minimum error in labeling
    
    %Calculate beta
    beta_t = eps_t / (1-eps_t);
    
    %Calculate trust factor for each classifier
    alpha(iter) = log(1/beta_t);
    h(4,iter) = alpha(iter);
        
    %Update weights
    beta_t_pow = beta_t.^(1-xor(label, h_res(:,iter)));
    weight = weight.*beta_t_pow;
    
    %Calculate strong classifier
    C_temp = h_res(:,1:iter) * alpha(1:iter,1);
    thresh = min(C_temp(1:nPos));
    for j = 1:nImg_total
        if C_temp(j) >= thresh
            %Car present
            strongClass_result(j) = 1;
        else
            %Car absent
            strongClass_result(j) = 0;
        end
    end
    
    %Calculation of accuracy
    strongClass_TP(iter) = sum(strongClass_result(1:nPos))/nPos;
    strongClass_FP(iter) = sum(strongClass_result(nPos+1:end))/nNeg;
    if ((strongClass_TP(iter) >= thresh_TP) && (strongClass_FP(iter) <= 0.5))
        break;
    end   
end
%Updating string classifier for cascade process
%Consider only samples which have FP for next stage
[sort_FP,sort_FP_idx] = sort(strongClass_result(nPos+1:end),'ascend');
if sum(sort_FP) > 0
    for k = 1:nNeg
        if sort_FP(k)>0
            FP_indx_left = sort_FP_idx(k:end);
            strongClass.updatedIdx = [1:nPos,FP_indx_left'+nPos];
            break;
        end
    end
else
    strongClass.updatedIdx = [1:nPos];
end
strongClass.ClassifierParams = h;
strongClass.numIterations = iter;
strongClass.FPRate = strongClass_FP(iter);
end
%%
%Build Weak Classifier and select the best weak classifier
function [bestWeakClassifier_minErr, bestWeakClassifier_p, bestWeakClassifier_feat, bestWeakClassifier_result, bestWeakClassifier_theta] = buildWeakClassifier(feat_all,nPos,nNeg,wt,img_label)
nFeat = size(feat_all,1); %number of features
nExamp = size(feat_all,2); %number of examples
Tplus = repmat(sum(wt(1:nPos)),[nExamp 1]);
Tneg = repmat(sum(wt(nPos+1:end)),[nExamp 1]);
bestWeakClassifier_minErr = Inf;

for i = 1:nFeat
%     feat_temp = feat_all(i,:);
    [sorted_Examp,sortIndx] = sort(feat_all(i,:),'ascend');
    
     %Sort weight and label 
     wt_sort = wt(sortIndx);
     label_sort = img_label(sortIndx);
     
     %Caclulate S+ and S- for each element in the sorted list
     Splus = cumsum(wt_sort.*label_sort);
     Sneg = cumsum(wt_sort) - Splus;
     
     error1 = Splus + (Tneg - Sneg);
     error2 = Sneg + (Tplus - Splus); 
     e = min(error1,error2);
     [min_e, min_e_idx] = min(e);
     classify_result = zeros(nExamp,1);
     
     %Classify the results
     if error1(min_e_idx) <= error2(min_e_idx) 
        p = -1;
        classify_result(min_e_idx + 1:end) = 1; 
        classify_result(sortIndx) = classify_result;
     else
        p = 1;
        classify_result(1:min_e_idx) = 1; 
        classify_result(sortIndx) = classify_result;
     end
     
     %Store the best weak classifier
     if min_e < bestWeakClassifier_minErr
         bestWeakClassifier_minErr = min_e;
         bestWeakClassifier_p = p;
         bestWeakClassifier_feat = i;
         bestWeakClassifier_result = classify_result;
         
         %Update Threshold
         if min_e_idx == 1
             bestWeakClassifier_theta = sorted_Examp(1) - 0.5;
         elseif min_e_idx == nFeat
             bestWeakClassifier_theta = sorted_Examp(nFeat) + 0.5;
         else
             bestWeakClassifier_theta = mean([sorted_Examp(min_e_idx), sorted_Examp(min_e_idx-1)]);
             %disp(['Min e index = ', num2str(min_e_idx)]);
             %bestWeakClassifier_theta = 0.5 * (sorted_Examp(min_e_idx) + sorted_Examp(min_e_idx-1));
         end %end threhold loop
     end %end weak classifier loop
end %end number of features loop
end
%%
%Prediction of test set
function [test_result] = predictionTestSet(feat_posSet, feat_negSet, strong_clf)
%Retrieve information for all weak classifiers constituting the strong classifier
nWeak = strong_clf.numIterations; 
h = strong_clf.ClassifierParams; 
ft = h(1,1:nWeak); %index of best feature 
thetat = h(2,1:nWeak); %threshold value for feature
pt = h(3,1:nWeak); %polarity 
alphat = h(4,1:nWeak); %trust factor for each weak classifier

feat_comb = [feat_posSet,feat_negSet]; %features x numb of images
nPos = size(feat_posSet,2);
nNeg = size(feat_negSet,2);
ntotImg = nPos + nNeg;
ht = zeros(ntotImg, nWeak);
H = zeros(ntotImg,1);
for iWeak = 1:nWeak
    for iImg = 1:ntotImg
        if (pt(iWeak) * feat_comb(ft(iWeak),iImg) <= pt(iWeak) * thetat(iWeak))
            ht(iImg,iWeak) = 1;
        end
    end
end

%Combine all the weak classifier results to get final strong classifier
alphat_ht = ht * alphat';
limit_H = 0.5 * sum(alphat);
for iImg = 1:ntotImg
    if alphat_ht(iImg) >= limit_H
        H(iImg) = 1;
    end
end
test_result.H = H;
test_result.nPos = nPos;
test_result.nNeg = nNeg;
end
