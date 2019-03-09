% % This function calculate the coordination of test data points in the
% % low-dimensional representation space based on the proposed mutual neighborhood
% % based criteria of entropy estimation.
function Test_Low_d_Y = FSLL_Test_DataPoint_LowDim_Reperesenter_func(Train_data,Test_data,K,LowDim_Train_data,Lambda,Entropy_val_flag, proposed_entropy)
% INPUTS ::
% Train_data : the train datap matrix as DxN_train.
% Test_data : the test data matrix as DxN_test.
% K : number of the neighbors
% LowDim_Train_data : the represented train data in the low-dimensional space.
% Lambda : the regularization coeeficient between SLEM and LLE neighborhood graph matrixes.
% Entropy_val_flag : it is a flag to determine the method of calculation of entropy value.
% proposed_entropy : the entropy value in empirical and Ln(K_Ave/K) based methods.

% OUTPUTS ::
% Test_Low_d_Y : matrix of test data points in the low-dimensional representation space.

[~,N_train] = size(Train_data);
[~,N_test] = size(Test_data);

for k1=1:N_test
    test_d = Test_data(:,k1);
    Temp_test_d = repmat(test_d,1,N_train);
    Temp_euclidian_dist = sum((Temp_test_d-Train_data).^2);
    [sort_dist, ind1] = sort(Temp_euclidian_dist);
    
    %% Calculation of K neighbors and relevant number of mutual neighbors for each data point
    X_t =[test_d Train_data];
    [D,N] = size(X_t);
    X2 = sum(X_t.^2,1);
    distance = repmat(X2,N,1)+repmat(X2',1,N)-2*X_t'*X_t;
    [~,index] = sort(distance);
    [~, num_neighboor] = Mutual_Neighboors_func(index,K,N);
    
    %% Calculating the SLEM coefficients
    test_data_distance = zeros(1,N_train);
    test_data_distance(ind1(1:K)) = sort_dist(1:K);
    
    Wt_SLEM = zeros(1, N_train);
    beta = 0.01;
    
    if (Entropy_val_flag==2)
        Entropy_val = log(num_neighboor/K); % Ln(ki/K)
    else
        Entropy_val = proposed_entropy; % Ln(K_Ave/K) entropy value or empirical entropy value
    end
    
    
    Temp_neighb = find(test_data_distance > 0);
    
    % Set minimum and maximum values for precision
    betamin = -Inf;
    betamax = Inf;
    
    % Compute the Gaussian kernel and entropy for the current precision
    [Entropy_Wt, SLEM_coeff] = Hbeta(test_data_distance, beta,Temp_neighb,N_train);
    
    % Evaluate whether the perplexity is within tolerance
    Entropy_diff = Entropy_Wt - Entropy_val;
    tries = 0;
    tol = 1e-5;
    while abs(Entropy_diff) > tol && tries < 50
        % If not, increase or decrease precision
        if Entropy_diff > 0
            betamin = beta;
            if isinf(betamax)
                beta = beta * 2;
            else
                beta = (beta + betamax) / 2;
            end
        else
            betamax = beta;
            if isinf(betamin)
                beta = beta / 2;
            else
                beta = (beta + betamin) / 2;
            end
        end
        
        % Recompute the values
        [Entropy_Wt, SLEM_coeff] = Hbeta(test_data_distance, beta,Temp_neighb,N_train);
        Entropy_diff = Entropy_Wt - Entropy_val;
        tries = tries + 1;
    end
    % Set the final row of P
    Wt_SLEM = SLEM_coeff(ind1(1:K));
    
    %% Calculation the LLE reconstruction coefficients
    if(K>D)
        tol=1e-3; % regularlizer in case constrained fits are ill conditioned
    else
        tol=0;
    end
    z = Train_data(:,ind1(1:K))-repmat(test_d,1,K); % shift ith pt to origin
    C = z'*z; % local covariance
    if rank(C)== K
        Wt_LLE = C\ones(K,1); % solve Cw=1
    else
        C = C + trace(C)*1e-6*eye(K);
        Wt_LLE = C\ones(K,1);
    end
    Wt_LLE = Wt_LLE/(max(sum(Wt_LLE),eps)); % enforce sum(w)=1
    
    %% Calculating the coordination of test data point in low-dimensional representation space
    Test_Low_d_Y(:,k1) = (LowDim_Train_data(:,ind1(1:K))*(Lambda*Wt_LLE+Wt_SLEM'))/(Lambda+sum(Wt_SLEM));
    
end