% Function that computes the Gaussian kernel values given a vector of
% squared Euclidean distances, and the precision of the Gaussian kernel.
% The function also computes the entropy of the distribution.
function [H, P] = Hbeta(D, beta,Temp_neighb,N)
P = zeros(1,N);
P(Temp_neighb) = exp(-D(Temp_neighb) * beta);
sumP = sum(P);
sumP = max(sumP,eps);
H = log(sumP) + beta * sum(D .* P) / sumP;
P = P / sumP;
end