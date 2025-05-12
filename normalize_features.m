% Author: John Connor Prikkel
% Date: 8/30/23
% ECE 595 Assignment 2

function [x_norm] = normalize_features(x)
% Normalizes the feature vector and returns the normalized feature vector 

% M = number of rows, n = number of columns
[M,n] = size(x);

mu = mean(x);
sigma = std(x);

% Use repmat to make mu the same size as x to subtract
mu = repmat(mu, [M, 1]);

x_norm = (x - mu)./sigma;

end