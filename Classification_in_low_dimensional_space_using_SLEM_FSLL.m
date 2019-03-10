%% Stochastic Laplacian Eigenmap (SLEM) & Fusion of LLE and SLEM (FSLL) manifold learning methods implementation
% It is created by R. Hajizadeh (PhD in electrical engineering)
% SLEM and FSLL methods are local Manifold Learning methods that we have proposed in the article with title:
% "Fusion of LLE and stochastic LEM for Persian handwritten digits recognition" in IJDAR journal (Link: https://link.springer.com/article/10.1007/s10032-018-0303-4).

%% initializing
close all;clear; clc;

Lambda=0.5; % the regularization coeeficient between SLEM and LLE neighborhood graph matrixes.
% if Lambda set zero (Lambda=0), just SLEM method is applied for dimension
% reduction. In the other word, if Lambda=0, this code is for SLEM
% implementation.

K = 10; % number of the neighbors
K_nn = 3; % K of proposed modified-kNN classifier

d_vec = [10 15 20 30 50 100]; % determining the dimensions in the low-dimensional representation space

% by setting the "Entropy_val_flag", the method of determining entropy value is selected:
% Entropy_val_flag = 1: empirical entropy value, (default)
% Entropy_val_flag = 2: Ln(ki / K) (mutual neighborhood based entropy estimation) 
% Entropy_val_flag = 3: Ln(K_Ave / K) (mutual neighborhood based entropy estimation), 
Entropy_val_flag = 1;

%% import high-dimensional input and its information such as: labels, number of classes and number of test datapoints.

% % The loaded matrices consist random selection of the Persian handwritten character images from HODA
% database (500 images of each class for train and 300 images for test) with their labels.
% HODA database include 32 different Persian handwritten character classes.
% Also, HMaX features of these images have been extracted (in 400-dimensional space).
% Therefore, we have a train data matrix as 401x16000 and a test data matrix as 401x9600 ((D+1)xN).

% load('HODA_labeled_Train_Test.mat', 'labled_Trian_chr_HODA_500')
% load('HODA_labeled_Train_Test.mat', 'labled_Test_chr_HODA_300')
% Train_label = labled_Trian_chr_HODA_500(end,:);
% Test_label = labled_Test_chr_HODA_300(end,:);
% Train_data = labled_Trian_chr_HODA_500(1:end-1,:);
% Test_data = labled_Test_chr_HODA_300(1:end-1,:);
% All_test_num = 9600;
% num_class = 18;


% % The loaded matrices consist random selection of the Persian handwritten digit images from FHT
% database (140 images of each class for train and 140 images for test).
% The database consists 280 Persian handwritten digit images from 1-9 (except 2).
% Each image has been vectorized. Therefore, we have a train and test data matrices as 4096x1120 (DxN).

load('Train_manifold_1120_FHT.mat', 'Train_manif')
Train_data = Train_manif;
load('Test_manifold_1120_FHT.mat', 'Test_manif')
Test_data = Test_manif;

preTrain_label = repmat(1:8,140,1);
Train_label = preTrain_label(:)';
preTest_label = repmat(1:8,140,1);
Test_label = preTest_label(:)';

All_test_num = 1120;
num_class = 8;
%%  STEP1: COMPUTE PAIRWISE DISTANCES & FIND NEIGHBORS(euclidian distance)
[D,N] = size(Train_data); % D: dimension of high-dimensional input data - N: number of the data points 

Temp_X = sum(Train_data.^2,1);
distances = repmat(Temp_X,N,1)+repmat(Temp_X',1,N)-2*Train_data'*Train_data;
[~,sorted_index] = sort(distances);
Neighbors_index = sorted_index(2:(1+K),:);

[Mutual_Neighbors, num_MN_neighbor] = Mutual_Neighboors_func(sorted_index,K,N); % determine mutual-neighbors of each input data point

%%  calculationg the LLE neighborhood graph matrix 
if(K>D)
    tol=1e-3; % regularlizer in case constrained fits are ill conditioned
else
    tol=0;
end
W_LLE = zeros(K,N);
for ii=1:N
    z = Train_data(:,Neighbors_index(:,ii))-repmat(Train_data(:,ii),1,K); % shift ith pt to origin
    C = z'*z;   % local covariance
    %     C = C + eye(K,K)*tol*trace(C);  % regularlization (K>D)
    W_LLE(:,ii) = C\ones(K,1); % solve Cw=1
    W_LLE(:,ii) = W_LLE(:,ii)/sum(W_LLE(:,ii)); % enforce sum(w)=1
end;

Temp_W = zeros(N);
for kk3 =1:N
    Temp_W(kk3,Neighbors_index(:,kk3))=W_LLE(:,kk3);
end
M_LLE = (eye(N)-Temp_W)'*(eye(N)-Temp_W);

%%  calculationg the SLEM neighborhood graph matrix
W_SLEM = zeros(N,N);
for k2=1:N
    W_SLEM(k2,Neighbors_index(:,k2)) = distances(k2,Neighbors_index(:,k2));
end

All_SLEM_Coeffs = zeros(N, N);
beta = 0.01 * ones(N, 1);

% selection of how determining the entropy value
if (Entropy_val_flag==1) % empirical entropy value
    Entropy_Value = 0.5; 
    
elseif (Entropy_val_flag==2) % ln(ki / K) criterion of mutual neighborhood based entropy estimation
    Entropy_Value_datapoints = num_MN_neighbor./K;
    
else (Entropy_val_flag==3) %  ln(K_Ave / K) criterion of mutual neighborhood based entropy estimation
    K_Ave = (sum(num_MN_neighbor)/N);
    Entropy_Value = log(K_Ave/K); 
    
end

for c1=1:N
    if (Entropy_val_flag==2)
        Entropy_Value = Entropy_Value_datapoints(c1);
    end
    Temp_neighb = find(W_SLEM(c1,:)>0);
    
    % Set minimum and maximum values for precision
    betamin = -Inf;
    betamax = Inf;
    
    % Compute the Gaussian kernel and entropy for the current precision
    [Entropy_W, SLEM_coeff] = Hbeta(W_SLEM(c1,:), beta(c1),Temp_neighb,N);
    
    % Evaluate whether the entropy value is within tolerance
    Entropy_diff = Entropy_W - Entropy_Value;
    tries = 0;
    tol = 1e-5;
    while abs(Entropy_diff) > tol && tries < 50
        % If not, increase or decrease precision
        if Entropy_diff > 0
            betamin = beta(c1);
            if isinf(betamax)
                beta(c1) = beta(c1) * 2;
            else
                beta(c1) = (beta(c1) + betamax) / 2;
            end
        else
            betamax = beta(c1);
            if isinf(betamin)
                beta(c1) = beta(c1) / 2;
            else
                beta(c1) = (beta(c1) + betamin) / 2;
            end
        end
        
        % Recompute the values
        [Entropy_W, SLEM_coeff] = Hbeta(W_SLEM(c1,:), beta(c1),Temp_neighb,N);
        Entropy_diff = Entropy_W - Entropy_Value;
        tries = tries + 1;
    end
    %Tot_Entropy_W(i) = Entropy_W;
    
    % Set the final row of P
    All_SLEM_Coeffs(c1, :) = SLEM_coeff;
end
W_SLEM = All_SLEM_Coeffs;
W_SLEM = max(W_SLEM,W_SLEM');

D_SLEM = diag(sum(W_SLEM')+sum(W_SLEM));
L_SLEM = D_SLEM - 2*W_SLEM;

%% Fusion of LLE and SLEM graph matrixes and calculating embedded data points in the low-dimensional space.
M_FSLL = L_SLEM + Lambda*M_LLE;

rank_M_FSLL = rank(M_FSLL);
temp_num = N - rank_M_FSLL;
[preY_train_FSLL, Eigenvals_M_FSLL] = eig(full(M_FSLL));
[~, Sorted_ind_EigVal_FSLL]= sort(abs(diag(Eigenvals_M_FSLL)));

% calculating embedded low-dimensional data points
for k_d=1:length(d_vec)
    d = d_vec(k_d);
    if rank_M_FSLL < d
        Y_train_FSLL = preY_train_FSLL(:,Sorted_ind_EigVal_FSLL((end-d+1):(end)))'*sqrt(N);
    else
        Y_train_FSLL = preY_train_FSLL(:,Sorted_ind_EigVal_FSLL((temp_num+1):(temp_num+d)))'*sqrt(N);
    end
 
    %  low-dimensional test datapoints calculator
    Y_test_FSLL = FSLL_Test_DataPoint_LowDim_Reperesenter_func(Train_data, Test_data, K, Y_train_FSLL, Lambda, Entropy_val_flag, Entropy_Value);
    
    % classification in the low-dimensional space
    labled_Y_train = [Y_train_FSLL; Train_label];
    labled_Y_test = [Y_test_FSLL; Test_label];
    [Predicted_lable{k_d}, Low_RR(k_d)] = Modified_kNN_classifier_func(num_class,labled_Y_train,labled_Y_test,All_test_num,K_nn)    
    
    figure(k_d)
    plotconfusion(Test_label,Predicted_lable{k_d})
end
