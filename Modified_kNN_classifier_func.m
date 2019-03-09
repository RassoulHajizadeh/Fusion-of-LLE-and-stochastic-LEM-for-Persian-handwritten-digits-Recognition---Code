%% This function is a new proposed modified-KNN classifier that is described and used in article with title: Fusion of LLE and stochastic LEM for Persian handwritten digits recognition
function [Test_class_lable, recognition_rate] = Modified_kNN_classifier_func(num_class,Train_data,Test_data,All_test_num,K_nn)

% INPUTS ::
% num_class : number of class of input data
% Train_data : the train matrix as (D+1)xN_train which consists train data ...
% "Train_data(1:(end-1),:)" and labels of each train data point "Train_data(end,:)".
% Test_data : the test matrix as (D+1)xN_tests which consists test data ...
% "Test_data(1:(end-1),:)" and labels of each test data point "Test_data(end,:)".
% All_test_num : number of all test data point (that is equal N_test).
% K_nn : is the number of K in KNN classifier.

% OUTPUTS ::
% recognition_rate : the percentage of the recognition rate
% Test_class_lable : the predicted label of each test data point (that is usefu for plot confusion matrix)

true_counter = 0;
false_counter = 0;

X =Train_data(1:(end-1),:);
Test_class_lable = zeros(num_class,All_test_num);

for k1=1:All_test_num
    
    Temp_test_d=Test_data(:,k1);
    Temp_test_lable = Temp_test_d(end);
    
    [r1,c1] = size(X);
    temp3 = repmat(Temp_test_d(1:end-1),1,c1);
    dist1 = (temp3 - X).^2;
    [sorted1,index1] = sort(sum(dist1));
    
    flags_of_classes = zeros(1,num_class);
    flags_of_Val_classes = zeros(1,num_class);
    for k5 = 1:K_nn
        flags_of_classes(Train_data(end,index1(k5))) = flags_of_classes(Train_data(end,index1(k5)))+1;
        flags_of_Val_classes(Train_data(end,index1(k5)))  = flags_of_Val_classes(Train_data(end,index1(k5))) + sorted1(k5);
    end
    [val_class_label, class_label] = max(flags_of_classes);
    if val_class_label > (K_nn/2)
        if  class_label == Temp_test_lable
            true_counter = true_counter +1;
        else
            false_counter = false_counter +1;
        end
    else
        sel_labels = find(flags_of_classes==val_class_label);
        if length(sel_labels) == 1
            if  class_label == Temp_test_lable
                true_counter = true_counter +1;
            else
                false_counter = false_counter +1;
            end
        else
            [val_max2, ind2] = min(flags_of_Val_classes(sel_labels));
            class_label = sel_labels(ind2);
            if  class_label == Temp_test_lable
                true_counter = true_counter +1;
            else
                false_counter = false_counter +1;
            end
        end
    end
    Test_class_lable(class_label,k1) = class_label;
end
recognition_rate = (true_counter/(true_counter+false_counter))*100;