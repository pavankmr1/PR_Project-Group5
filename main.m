%binary classification on drug consumption dataset
% User=1 and Non-User=0
clear all; clc; close all; 

%------------------Load the Dataset------------------------------------%
drugUserData = load(['drug_consumption.txt']);

%select all features for classification
X=drugUserData(:, 2:13);

%-----------All drug Names-----------------------%
drugNames={'alcohol','amphet','amyl','benzos','caff','cannabis','choc','coke','crack','ecstacy','heroin','ketamine','legalh','LSD','meth','mushroom','nicotine','VSA'};

%-----------All classification Model Names-----------------------%
classifierName={'KNN Classifier k-3','KNN Classifier k-5','KNN Classifier k-7','Decision Tree','Linear SVM Model'};

%store the results in a table like model
%name,accuracy,sensitivity,specificity,NumberOfFolds
%individual tables for each classifier

Results1 = table;Results2 = table;Results3 = table;Results4 = table;
Results5 = table;Results6 = table;Results7 = table;Results8 = table;
Results9 = table;Results10 = table;Results11 = table;Results12 = table;
ResultsAlcohol=table;

%Save all drugs reults in a file
allDrugsResults=table;
sensitivity=0;specificity=0;

drugNumber=13;

for j=1:18
    if (j==18)
        drugNumber=drugNumber+2;
    else
        drugNumber=drugNumber+1;
    end
    %Display the Drug Name.
    disp('Drug Name:')
    disp(drugNames(j));
    Y=drugUserData(:,drugNumber);
%---------------------K fold Cross validation------------------------%
% used 2 folds 5 and 10
for foldSize= 5:5:10
c = cvpartition(Y,'KFold',foldSize);
%--------------------------------------------------------------------%
%for both the folds divide the dataset into training and testing set
for i=1:foldSize
    %train data
    Dtrain = X(training(c,i),:);
    %train labels
    Ltrain = Y(training(c,i));
    %tset data
    Dtest=X(test(c,i),:);
    %test labels
    Ltest=Y(test(c,i));

    %Prediction using all classification Models
    %for each model calculate accuracy, sensitivity and specificity
    %------------------------------------------------------------------------------------------------------
    %------------------------------------------------------------------------------------------------------
    %-----------KNN classifier k=3----------------------------------------%
    KNN_Model = fitcknn(Dtrain, Ltrain,'NumNeighbors',3);
    [accuracyK3,sensitivityK3,specificityK3]=predModel(KNN_Model,Dtest,Ltest);
    ResultsK3 = table(classifierName(2), foldSize, accuracyK3,sensitivityK3,specificityK3);
    Results2=[Results2;ResultsK3];
    
    %------------------------------------------------------------------------------------------------------
    %---------------KNN classifier k=5------------------------------------%
    KNN_Model = fitcknn(Dtrain, Ltrain,'NumNeighbors',5);
    [accuracyK5,sensitivityK5,specificityK5]=predModel(KNN_Model,Dtest,Ltest);
    ResultsK5 = table(classifierName(3), foldSize, accuracyK5,sensitivityK5,specificityK5);
    Results3=[Results3;ResultsK5];

    %------------------------------------------------------------------------------------------------------
    %----------------KNN classifier k=7
    KNN_Model = fitcknn(Dtrain, Ltrain,'NumNeighbors',7);
    [accuracyK7,sensitivityK7,specificityK7]=predModel(KNN_Model,Dtest,Ltest);
    ResultsK7 = table(classifierName(4), foldSize, accuracyK7,sensitivityK7,specificityK7);
    Results4=[Results4;ResultsK7];
    
    %------------------------------------------------------------------------------------------------------
    %-----------------Decision Tree---------------------------------------%
    DTT_Model = fitctree(Dtrain, Ltrain);
    [accuracyDT,sensitivityDT,specificityDT]=predModel(DTT_Model,Dtest,Ltest);
    ResultsDT = table(classifierName(5), foldSize, accuracyDT,sensitivityDT,specificityDT);
    Results5=[Results5;ResultsDT];

    %------------------------------------------------------------------------------------------------------
    %------------------SVM Linear SVM Model-------------------------------%
    SVM_Model1 = fitcsvm(Dtrain, Ltrain,'KernelFunction','linear', 'BoxConstraint', 10);
    [accuracySVM2,sensitivitySVM2,specificitySVM2]=predModel(SVM_Model1,Dtest,Ltest);
    ResultsSVM2 = table(classifierName(5), foldSize, accuracySVM2,sensitivitySVM2,specificitySVM2);
    Results9=[Results9;ResultsSVM2];
end
%------------------------------------------------------------------------%
%----------Save the mean accuracy for both folds-------------------------%
%for each classifier.

ResultsAlcohol2=table(classifierName(2), foldSize, mean(accuracyK3),mean(sensitivityK3),mean(specificityK3));
ResultsAlcohol=[ResultsAlcohol;ResultsAlcohol2];

ResultsAlcohol3=table(classifierName(3), foldSize, mean(accuracyK5),mean(sensitivityK5),mean(specificityK5));
ResultsAlcohol=[ResultsAlcohol;ResultsAlcohol3];

ResultsAlcohol4=table(classifierName(4), foldSize, mean(accuracyK7),mean(sensitivityK7),mean(specificityK7));
ResultsAlcohol=[ResultsAlcohol;ResultsAlcohol4];

ResultsAlcohol5=table(classifierName(5), foldSize, mean(accuracyDT),mean(sensitivityDT),mean(specificityDT));
ResultsAlcohol=[ResultsAlcohol;ResultsAlcohol5];

ResultsAlcohol9=table(classifierName(5), foldSize, mean(accuracySVM2),mean(sensitivitySVM2),mean(specificitySVM2));
ResultsAlcohol=[ResultsAlcohol;ResultsAlcohol9];

%--------------clear the tables to reuse--------------------------------%
Results1 = table;Results2 = table;Results3 = table;Results4 = table;
Results5 = table;Results6 = table;Results7 = table;Results8 = table;
Results9 = table;Results10 = table;Results11 = table;Results12 = table;

end

%------------------------------------------------------------------------%
%Display the table with all classifiers and both folds

%Change table variable name to understand it better
%ResultsAlcohol.Properties.VariableNames ={'Classifier Name','Fold Size','Accuracy','Sensitivity','Sensitivity'};
ResultsAlcohol.Properties.VariableNames{'Var3'} = 'accuracy';
ResultsAlcohol.Properties.VariableNames{'Var4'} = 'sensitivity';
ResultsAlcohol.Properties.VariableNames{'Var5'} = 'specificity';
disp(ResultsAlcohol);
allDrugsResults=[allDrugsResults;ResultsAlcohol];
%-------------------------------------------------------------------------%
%-----Find the classifier with maximum accuracy--------------------------%
[~,maxClassifierAccuracy]=max(ResultsAlcohol.accuracy);
r=ResultsAlcohol(maxClassifierAccuracy,:);
disp("Classifier with Maximum Accuracy is: ");
disp(r);
%------------------------------------------------------------------------%

%plot the results
figure;
bar(ResultsAlcohol.accuracy,0.4);
title({'Accuracy with all Classification Models'});
ylabel('Accuracy');
xlabel({'Fold:5                                                    Fold 10' ;'\fontsize{15}Classification Models'});
xlim=([0 16]);
xticks([1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24]);
xtickangle(45);
xticklabels({'KNN Classifier k-3','KNN Classifier k-5','KNN Classifier k-7','Decision Tree','Linear SVM','KNN Classifier k-3','KNN Classifier k-5','KNN Classifier k-7','Decision Tree','Linear SVM'});

ResultsAlcohol=table;
end
%-------------------------------------------------------------------------%
%Save the results in a file
filename='allDrugsResults';
writetable( allDrugsResults, [ filename '.csv'], 'Delimiter', ',', 'WriteRowNames', true);

%----------Function to Calculate
%accuracy,sensitivity,specificity------------------------%
function [accuracy,sensitivity,specificity]=predModel(model,Dtest,Ltest)
Lpred = model.predict(Dtest);
[conf, classorder] = confusionmat(Ltest,Lpred);
accuracy=trace(conf)/sum(conf(:));
idx1 = find(Ltest==0); 
pred1 = Lpred(idx1); 
sensitivity = length(find(pred1==0))/length(idx1); 
idx2 = find(Ltest==1); 
pred2 = Lpred(idx2); 
specificity = length(find(pred2==1))/length(idx2);
end